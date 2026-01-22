from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime
class AXNodeId(str):
    """
    Unique accessibility node identifier.
    """

    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> AXNodeId:
        return cls(json)

    def __repr__(self):
        return 'AXNodeId({})'.format(super().__repr__())