from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
@event_class('Page.loadEventFired')
@dataclass
class LoadEventFired:
    timestamp: network.MonotonicTime

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> LoadEventFired:
        return cls(timestamp=network.MonotonicTime.from_json(json['timestamp']))