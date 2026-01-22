from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
class MouseButton(enum.Enum):
    NONE = 'none'
    LEFT = 'left'
    MIDDLE = 'middle'
    RIGHT = 'right'
    BACK = 'back'
    FORWARD = 'forward'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)