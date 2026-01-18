from typing import Any, Dict, List
import pytest
import panel as pn
from panel.widgets import TextToSpeech, Utterance, Voice
@pytest.fixture
def voices():
    return Voice.to_voices_list(_VOICES_FIREFOX_WIN10)