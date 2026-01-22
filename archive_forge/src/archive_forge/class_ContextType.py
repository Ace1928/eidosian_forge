from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
class ContextType(enum.Enum):
    """
    Enum of BaseAudioContext types
    """
    REALTIME = 'realtime'
    OFFLINE = 'offline'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)