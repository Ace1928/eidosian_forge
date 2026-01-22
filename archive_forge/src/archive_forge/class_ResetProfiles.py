from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
@event_class('HeapProfiler.resetProfiles')
@dataclass
class ResetProfiles:

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ResetProfiles:
        return cls()