from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import target
class PermissionSetting(enum.Enum):
    GRANTED = 'granted'
    DENIED = 'denied'
    PROMPT = 'prompt'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)