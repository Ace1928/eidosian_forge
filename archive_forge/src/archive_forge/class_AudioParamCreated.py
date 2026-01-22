from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('WebAudio.audioParamCreated')
@dataclass
class AudioParamCreated:
    """
    Notifies that a new AudioParam has been created.
    """
    param: AudioParam

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AudioParamCreated:
        return cls(param=AudioParam.from_json(json['param']))