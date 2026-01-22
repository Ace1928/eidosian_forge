from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
class CacheId(str):
    """
    Unique identifier of the Cache object.
    """

    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> CacheId:
        return cls(json)

    def __repr__(self):
        return 'CacheId({})'.format(super().__repr__())