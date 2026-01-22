from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
class NodeId(int):
    """
    Unique DOM node identifier.
    """

    def to_json(self) -> int:
        return self

    @classmethod
    def from_json(cls, json: int) -> NodeId:
        return cls(json)

    def __repr__(self):
        return 'NodeId({})'.format(super().__repr__())