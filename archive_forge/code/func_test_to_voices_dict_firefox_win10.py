from typing import Any, Dict, List
import pytest
import panel as pn
from panel.widgets import TextToSpeech, Utterance, Voice
def test_to_voices_dict_firefox_win10():
    voices = Voice.to_voices_list(_VOICES_FIREFOX_WIN10)
    actual = Voice.group_by_lang(voices)
    assert 'en-US' in actual
    assert len(actual['en-US']) == 2