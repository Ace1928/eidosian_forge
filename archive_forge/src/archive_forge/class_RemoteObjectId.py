from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
class RemoteObjectId(str):
    """
    Unique object identifier.
    """

    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> RemoteObjectId:
        return cls(json)

    def __repr__(self):
        return 'RemoteObjectId({})'.format(super().__repr__())