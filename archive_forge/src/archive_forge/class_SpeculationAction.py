from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
class SpeculationAction(enum.Enum):
    """
    The type of preloading attempted. It corresponds to
    mojom::SpeculationAction (although PrefetchWithSubresources is omitted as it
    isn't being used by clients).
    """
    PREFETCH = 'Prefetch'
    PRERENDER = 'Prerender'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)