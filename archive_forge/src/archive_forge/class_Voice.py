from __future__ import annotations
import uuid
from typing import (
import param
from panel.widgets import Widget
from ..models.text_to_speech import TextToSpeech as _BkTextToSpeech
class Voice(param.Parameterized):
    """
    The current device (i.e. OS and Browser) provides a list of
    Voices. Each with a unique name and speaking a specific language.

    Wraps the HTML5 SpeecSynthesisVoice API

    See https://developer.mozilla.org/en-US/docs/Web/API/SpeechSynthesisVoice
    """
    default = param.Boolean(constant=True, default=False, doc='\n        A Boolean indicating whether the voice is the default voice\n        for the current app language (True), or not (False.)')
    lang = param.String(constant=True, doc='\n        Returns a BCP 47 language tag indicating the language of the voice.')
    local_service = param.Boolean(constant=True, doc='\n        A Boolean indicating whether the voice is supplied by a local\n        speech synthesizer service (True), or a remote speech\n        synthesizer service (False.)')
    name = param.String(constant=True, doc='\n        Returns a human-readable name that represents the voice.')
    voice_uri = param.String(constant=True, doc='\n        Returns the type of URI and location of the speech synthesis\n        service for this voice.')

    @staticmethod
    def to_voices_list(voices):
        """Returns a list of Voice objects from the list of dicts provided"""
        result = []
        for _voice in voices:
            voice = Voice(**_voice)
            result.append(voice)
        return result

    @staticmethod
    def group_by_lang(voices):
        """Returns a dictionary where the key is the `lang` and the value is a list of voices
        for that language."""
        if not voices:
            return {}
        sorted_lang = sorted(list(set((voice.lang for voice in voices))))
        result = {lang: [] for lang in sorted_lang}
        for voice in voices:
            result[voice.lang].append(voice)
        result = {key: sorted(value, key=lambda x: x.name) for key, value in result.items()}
        return result