from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
class LogicalAxes(enum.Enum):
    """
    ContainerSelector logical axes
    """
    INLINE = 'Inline'
    BLOCK = 'Block'
    BOTH = 'Both'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)