from __future__ import annotations
import uuid
from typing import (
import param
from panel.widgets import Widget
from ..models.text_to_speech import TextToSpeech as _BkTextToSpeech
class TextToSpeech(Utterance, Widget):
    """
    The `TextToSpeech` widget wraps the HTML5 SpeechSynthesis API

    See https://developer.mozilla.org/en-US/docs/Web/API/SpeechSynthesis

    Reference: https://panel.holoviz.org/reference/widgets/TextToSpeech.html

    :Example:

    >>> TextToSpeech(name="Speech Synthesis", value="Data apps are nice")
    """
    auto_speak = param.Boolean(default=True, doc='\n        Whether or not to automatically speak when the value changes.')
    cancel = param.Event(doc='\n        Removes all utterances from the utterance queue.')
    pause = param.Event(doc='\n        Puts the TextToSpeak object into a paused state.')
    resume = param.Event(doc='\n        Puts the TextToSpeak object into a non-paused state: resumes\n        it if it was already paused.')
    paused = param.Boolean(readonly=True, doc='\n        A Boolean that returns true if the TextToSpeak object is in a\n        paused state.')
    pending = param.Boolean(readonly=True, doc='\n        A Boolean that returns true if the utterance queue contains\n        as-yet-unspoken utterances.')
    speak = param.Event(doc='\n        Speak. I.e. send a new Utterance to the browser')
    speaking = param.Boolean(readonly=True, doc='\n        A Boolean that returns true if an utterance is currently in\n        the process of being spoken — even if TextToSpeak is in a\n        paused state.')
    voices = param.List(readonly=True, doc='\n        Returns a list of Voice objects representing all the available\n        voices on the current device.')
    _voices = param.List()
    _rename: ClassVar[Mapping[str, str | None]] = {'auto_speak': None, 'lang': None, 'name': None, 'pitch': None, 'rate': None, 'speak': None, 'value': None, 'voice': None, 'voices': None, 'volume': None, '_voices': 'voices'}
    _widget_type: ClassVar[Type[Model]] = _BkTextToSpeech

    def _process_param_change(self, msg):
        speak = msg.get('speak') or ('value' in msg and self.auto_speak)
        msg = super()._process_param_change(msg)
        if speak:
            msg['speak'] = self.to_dict()
        return msg

    @param.depends('_voices', watch=True)
    def _update_voices(self):
        voices = []
        for _voice in self._voices:
            voice = Voice(**_voice)
            voices.append(voice)
        self.voices = voices
        self.set_voices(self.voices)

    def __repr__(self, depth=None):
        return f'TextToSpeech(name={self.name!r})'

    def __str__(self):
        return f'TextToSpeech(name={self.name!r})'