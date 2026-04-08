"""
Tests for Phase 1: Per-Person Real-Time Coaching Signal Chain.

Coverage:
  - _resolve_speaker_name: valid indices, out-of-range, empty list, direct speaker_id match
  - System prompt includes "Always name the specific person"
  - ELM prompt includes counterpart name when available
  - General prompt uses resolved names in transcript section
  - User archetype auto-detection mid-session
  - Session debrief includes per-participant data
  - Retro debrief has same 5-section structure
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.coaching_engine import CoachingEngine, CoachingPrompt, _SYSTEM_PROMPT
from backend.elm_detector import ELMEvent
from backend.profiler import (
    ParticipantProfiler,
    UserBehaviorObserver,
    WindowClassification,
    _aggregate_signals,
    _PROFILER_NEUTRAL_BAND,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RESPONSE_TEXT = "Sarah's defensive — ask what data would change her mind."


def make_mock_client(text: str = _RESPONSE_TEXT) -> Any:
    content = MagicMock()
    content.text = text
    response = MagicMock()
    response.content = [content]
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)
    return client


def make_engine(
    client: Any = None,
    user_archetype: str = "Firestarter",
    participants: list[dict] | None = None,
) -> CoachingEngine:
    return CoachingEngine(
        user_speaker="user",
        anthropic_client=client or make_mock_client(),
        elm_cadence_floor_s=0.0,
        general_cadence_floor_s=0.0,
        haiku_timeout_s=999.0,
        user_archetype=user_archetype,
        participants=participants,
    )


# ---------------------------------------------------------------------------
# _resolve_speaker_name tests
# ---------------------------------------------------------------------------

class TestResolveSpeakerName:
    """Tests for CoachingEngine._resolve_speaker_name."""

    def test_empty_participants_returns_empty(self):
        engine = make_engine(participants=[])
        assert engine._resolve_speaker_name("counterpart_0") == ""

    def test_none_participants_returns_empty(self):
        engine = make_engine(participants=None)
        assert engine._resolve_speaker_name("counterpart_0") == ""

    def test_counterpart_index_lookup(self):
        engine = make_engine(participants=[
            {"name": "Sarah Chen", "archetype": "Architect"},
            {"name": "Mike Johnson", "archetype": "Firestarter"},
        ])
        assert engine._resolve_speaker_name("counterpart_0") == "Sarah Chen"
        assert engine._resolve_speaker_name("counterpart_1") == "Mike Johnson"

    def test_speaker_index_lookup(self):
        """speaker_1 maps to participants[0] (speaker_0 is user)."""
        engine = make_engine(participants=[
            {"name": "Sarah Chen", "archetype": "Architect"},
        ])
        assert engine._resolve_speaker_name("speaker_1") == "Sarah Chen"

    def test_out_of_range_returns_empty(self):
        engine = make_engine(participants=[
            {"name": "Sarah Chen", "archetype": "Architect"},
        ])
        assert engine._resolve_speaker_name("counterpart_5") == ""

    def test_invalid_format_returns_empty(self):
        engine = make_engine(participants=[
            {"name": "Sarah Chen", "archetype": "Architect"},
        ])
        assert engine._resolve_speaker_name("unknown_speaker") == ""

    def test_direct_speaker_id_match(self):
        engine = make_engine(participants=[
            {"name": "Sarah Chen", "archetype": "Architect", "speaker_id": "counterpart_0"},
        ])
        assert engine._resolve_speaker_name("counterpart_0") == "Sarah Chen"

    def test_missing_name_field_returns_empty(self):
        engine = make_engine(participants=[
            {"archetype": "Architect"},
        ])
        assert engine._resolve_speaker_name("counterpart_0") == ""


# ---------------------------------------------------------------------------
# System prompt tests
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    """Verify system prompt instructs Haiku to name specific people."""

    def test_system_prompt_names_instruction(self):
        assert "Always name the specific person" in _SYSTEM_PROMPT

    def test_system_prompt_example_uses_name(self):
        assert "Sarah" in _SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# ELM prompt includes counterpart name
# ---------------------------------------------------------------------------

class TestELMPromptWithName:
    """Verify ELM prompts include counterpart name when available."""

    @pytest.mark.asyncio
    async def test_elm_prompt_includes_name(self):
        client = make_mock_client()
        engine = make_engine(
            client=client,
            user_archetype="Firestarter",
            participants=[
                {"name": "Sarah Chen", "archetype": "Architect"},
            ],
        )
        event = ELMEvent(
            speaker_id="counterpart_0",
            state="ego_threat",
            utterance="I disagree completely",
            evidence=["I disagree completely"],
        )
        profile = WindowClassification(
            speaker_id="counterpart_0",
            superpower="Architect",
            confidence=0.7,
            focus_score=50.0,
            stance_score=-30.0,
            utterance_count=5,
        )
        await engine.process(
            elm_event=event,
            participant_profile=profile,
            user_is_speaking=False,
        )
        # Check the user message sent to Haiku includes "Sarah Chen"
        call_args = client.messages.create.call_args
        user_msg = call_args.kwargs["messages"][0]["content"]
        assert "Sarah Chen" in user_msg
        assert "Sarah Chen (Architect)" in user_msg

    @pytest.mark.asyncio
    async def test_elm_prompt_without_name_still_works(self):
        client = make_mock_client()
        engine = make_engine(client=client, participants=[])
        event = ELMEvent(
            speaker_id="counterpart_0",
            state="ego_threat",
            utterance="I disagree completely",
            evidence=["I disagree completely"],
        )
        await engine.process(elm_event=event, user_is_speaking=False)
        call_args = client.messages.create.call_args
        user_msg = call_args.kwargs["messages"][0]["content"]
        assert "the counterpart" in user_msg


# ---------------------------------------------------------------------------
# General prompt uses resolved names in transcript
# ---------------------------------------------------------------------------

class TestGeneralPromptNames:
    """Verify _general_prompt uses resolved names in the transcript section."""

    @pytest.mark.asyncio
    async def test_transcript_labels_use_resolved_names(self):
        client = make_mock_client()
        engine = make_engine(
            client=client,
            participants=[
                {"name": "Sarah Chen", "archetype": "Architect"},
            ],
        )
        transcript = [
            {"speaker": "user", "text": "Let me explain the plan"},
            {"speaker": "counterpart_0", "text": "I need to see the data first"},
        ]
        await engine.process(
            recent_transcript=transcript,
            user_is_speaking=False,
        )
        call_args = client.messages.create.call_args
        user_msg = call_args.kwargs["messages"][0]["content"]
        assert "Sarah Chen: I need to see the data first" in user_msg
        assert "You: Let me explain the plan" in user_msg
        # Should NOT contain raw speaker IDs
        assert "counterpart_0" not in user_msg


# ---------------------------------------------------------------------------
# User archetype auto-detection
# ---------------------------------------------------------------------------

class TestUserArchetypeAutoDetection:
    """Test mid-session user archetype classification."""

    def test_aggregate_signals_produces_archetype(self):
        """Verify the profiler's signal aggregation can classify from accumulated data."""
        from backend.profiler import _score_utterance

        # Feed data-driven utterances (Logic + Analysis = Architect)
        texts = [
            "The data shows a 15% increase in conversion",
            "Based on the metrics and our benchmark analysis",
            "What does the research tell us about this hypothesis?",
            "I'd like to understand the correlation between these variables",
            "Looking at the chart, the trend is specifically downward",
        ]
        signals = [_score_utterance(t) for t in texts]
        focus, stance, confidence = _aggregate_signals(signals)
        # Logic-heavy signals should give positive focus (Architect/Inquisitor territory)
        assert focus > _PROFILER_NEUTRAL_BAND

    def test_update_user_archetype_changes_engine(self):
        """Integration: SessionPipeline._update_user_archetype updates CoachingEngine."""
        # Import here to avoid circular issues
        from backend.main import SessionPipeline

        engine = make_engine(user_archetype="Unknown")
        pipeline = SessionPipeline(
            session_id="test-session",
            user_id="test-user",
            user_speaker="user",
            coaching_engine=engine,
        )

        # Feed data-heavy utterances as the user
        data_utterances = [
            "The data shows 15% improvement in our metrics",
            "Based on our benchmark analysis, the evidence is clear",
            "What does the research suggest about this hypothesis?",
            "I'd like to understand the statistical correlation here",
            "Looking at the chart specifically, the trend is measurably downward",
        ]
        for text in data_utterances:
            pipeline.observer.add_utterance("user", text)

        # Trigger the auto-detection
        pipeline._update_user_archetype()

        # Should have classified to something other than "Unknown"
        assert engine._user_archetype != "Unknown"
        # Logic-heavy + analysis-heavy signals → Architect
        assert engine._user_archetype == "Architect"


# ---------------------------------------------------------------------------
# Session debrief includes per-participant data
# ---------------------------------------------------------------------------

class TestSessionDebriefParticipants:
    """Verify _generate_session_debrief includes per-participant profiles in prompt."""

    @pytest.mark.asyncio
    async def test_debrief_prompt_includes_participant_profiles(self):
        """When participants are provided, the Opus prompt should name each one."""
        from backend.main import _generate_session_debrief

        utterances = [
            {"speaker": "user", "text": "Let me walk you through the proposal"},
            {"speaker": "counterpart_0", "text": "I need to see the numbers before I commit"},
            {"speaker": "counterpart_1", "text": "This is exciting, let's move fast"},
            {"speaker": "user", "text": "The data shows 20% growth"},
        ]
        scores = {
            "persuasion_score": 72,
            "timing_score": 22,
            "ego_safety_score": 25,
            "convergence_score": 25,
        }
        participants = [
            {
                "speaker_id": "counterpart_0",
                "name": "Sarah Chen",
                "archetype": "Architect",
                "confidence": 0.75,
                "focus_score": 60.0,
                "stance_score": -40.0,
                "utterance_count": 8,
                "key_evidence": [{"text": "I need to see the numbers", "signals": {}, "strength": 3}],
                "elm_episodes": ["ego_threat"],
            },
            {
                "speaker_id": "counterpart_1",
                "name": "Mike Johnson",
                "archetype": "Firestarter",
                "confidence": 0.65,
                "focus_score": -50.0,
                "stance_score": 45.0,
                "utterance_count": 6,
                "key_evidence": [{"text": "This is exciting", "signals": {}, "strength": 2}],
                "elm_episodes": [],
            },
        ]

        # Mock the Anthropic client
        with patch("backend.main._anthropic.AsyncAnthropic") as mock_cls:
            content = MagicMock()
            content.text = "PARTICIPANT MAP\nSarah Chen (Architect)..."
            response = MagicMock()
            response.content = [content]
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=response)
            mock_cls.return_value = mock_client

            with patch("backend.main._load_settings", return_value={"anthropic_api_key": "test-key"}):
                with patch("backend.main.get_db_session") as mock_db:
                    mock_session = AsyncMock()
                    mock_session.get = AsyncMock(return_value=MagicMock())
                    mock_db.return_value.__aenter__ = AsyncMock(return_value=mock_session)
                    mock_db.return_value.__aexit__ = AsyncMock(return_value=None)

                    await _generate_session_debrief(
                        "test-session", utterances, scores,
                        participants=participants,
                        user_archetype="Inquisitor",
                    )

            # Verify the prompt sent to Opus includes participant names
            call_args = mock_client.messages.create.call_args
            prompt = call_args.kwargs["messages"][0]["content"]
            assert "Sarah Chen" in prompt
            assert "Mike Johnson" in prompt
            assert "Architect" in prompt
            assert "Firestarter" in prompt
            # Should have 5-section structure
            assert "PARTICIPANT MAP" in prompt
            assert "INTERACTION ANALYSIS" in prompt
            assert "KEY MOMENTS" in prompt
            assert "WHAT YOU DID WELL" in prompt
            assert "STRATEGIC PLAYBOOK" in prompt
            # Should include user archetype
            assert "Inquisitor" in prompt
            # Should include ELM episode data
            assert "ego_threat" in prompt

    @pytest.mark.asyncio
    async def test_debrief_without_participants_still_works(self):
        """Debrief should work when no participant profiles are available."""
        from backend.main import _generate_session_debrief

        utterances = [
            {"speaker": "user", "text": "Hello"},
            {"speaker": "counterpart_0", "text": "Hi there"},
        ]
        scores = {
            "persuasion_score": 50,
            "timing_score": 15,
            "ego_safety_score": 15,
            "convergence_score": 20,
        }

        with patch("backend.main._anthropic.AsyncAnthropic") as mock_cls:
            content = MagicMock()
            content.text = "Brief debrief."
            response = MagicMock()
            response.content = [content]
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=response)
            mock_cls.return_value = mock_client

            with patch("backend.main._load_settings", return_value={"anthropic_api_key": "test-key"}):
                with patch("backend.main.get_db_session") as mock_db:
                    mock_session = AsyncMock()
                    mock_session.get = AsyncMock(return_value=MagicMock())
                    mock_db.return_value.__aenter__ = AsyncMock(return_value=mock_session)
                    mock_db.return_value.__aexit__ = AsyncMock(return_value=None)

                    # Should not raise even without participants
                    await _generate_session_debrief(
                        "test-session", utterances, scores,
                    )

            # Verify it still called Opus
            assert mock_client.messages.create.called


# ---------------------------------------------------------------------------
# Debrief reinforcement — "What You Did Well"
# ---------------------------------------------------------------------------

class TestDebriefReinforcement:
    """Verify the debrief prompt asks for reinforcement of good practices."""

    @pytest.mark.asyncio
    async def test_debrief_prompt_asks_for_reinforcement(self):
        from backend.main import _generate_session_debrief

        utterances = [{"speaker": "user", "text": "Test"}]
        scores = {"persuasion_score": 70, "timing_score": 20, "ego_safety_score": 20, "convergence_score": 30}

        with patch("backend.main._anthropic.AsyncAnthropic") as mock_cls:
            content = MagicMock()
            content.text = "Good job."
            response = MagicMock()
            response.content = [content]
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=response)
            mock_cls.return_value = mock_client

            with patch("backend.main._load_settings", return_value={"anthropic_api_key": "k"}):
                with patch("backend.main.get_db_session") as mock_db:
                    mock_session = AsyncMock()
                    mock_session.get = AsyncMock(return_value=MagicMock())
                    mock_db.return_value.__aenter__ = AsyncMock(return_value=mock_session)
                    mock_db.return_value.__aexit__ = AsyncMock(return_value=None)

                    await _generate_session_debrief("s1", utterances, scores)

            prompt = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
            # Prompt should explicitly ask for reinforcement of good practices
            assert "WHAT YOU DID WELL" in prompt
            assert "Reinforce good practices" in prompt
