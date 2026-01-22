from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
class ContextState(enum.Enum):
    """
    Enum of AudioContextState from the spec
    """
    SUSPENDED = 'suspended'
    RUNNING = 'running'
    CLOSED = 'closed'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)