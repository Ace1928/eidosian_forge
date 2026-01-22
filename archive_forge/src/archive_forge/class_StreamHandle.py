from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
class StreamHandle(str):
    """
    This is either obtained from another method or specifed as ``blob:&lt;uuid&gt;`` where
    ``&lt;uuid&gt`` is an UUID of a Blob.
    """

    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> StreamHandle:
        return cls(json)

    def __repr__(self):
        return 'StreamHandle({})'.format(super().__repr__())