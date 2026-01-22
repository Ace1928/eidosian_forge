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
class FrameId(str):
    """
    Unique frame identifier.
    """

    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> FrameId:
        return cls(json)

    def __repr__(self):
        return 'FrameId({})'.format(super().__repr__())