from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import dom_debugger
from . import page
class StringIndex(int):
    """
    Index of the string in the strings table.
    """

    def to_json(self) -> int:
        return self

    @classmethod
    def from_json(cls, json: int) -> StringIndex:
        return cls(json)

    def __repr__(self):
        return 'StringIndex({})'.format(super().__repr__())