from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('WebAudio.audioListenerCreated')
@dataclass
class AudioListenerCreated:
    """
    Notifies that the construction of an AudioListener has finished.
    """
    listener: AudioListener

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AudioListenerCreated:
        return cls(listener=AudioListener.from_json(json['listener']))