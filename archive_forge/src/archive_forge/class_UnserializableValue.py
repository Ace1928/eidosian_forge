from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
class UnserializableValue(str):
    """
    Primitive value which cannot be JSON-stringified. Includes values ``-0``, ``NaN``, ``Infinity``,
    ``-Infinity``, and bigint literals.
    """

    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> UnserializableValue:
        return cls(json)

    def __repr__(self):
        return 'UnserializableValue({})'.format(super().__repr__())