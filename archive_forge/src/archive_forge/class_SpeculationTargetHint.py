from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
class SpeculationTargetHint(enum.Enum):
    """
    Corresponds to mojom::SpeculationTargetHint.
    See https://github.com/WICG/nav-speculation/blob/main/triggers.md#window-name-targeting-hints
    """
    BLANK = 'Blank'
    SELF = 'Self'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)