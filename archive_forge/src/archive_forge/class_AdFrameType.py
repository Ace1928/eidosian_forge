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
class AdFrameType(enum.Enum):
    """
    Indicates whether a frame has been identified as an ad.
    """
    NONE = 'none'
    CHILD = 'child'
    ROOT = 'root'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)