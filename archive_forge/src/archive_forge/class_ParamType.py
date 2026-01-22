from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
class ParamType(str):
    """
    Enum of AudioParam types
    """

    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> ParamType:
        return cls(json)

    def __repr__(self):
        return 'ParamType({})'.format(super().__repr__())