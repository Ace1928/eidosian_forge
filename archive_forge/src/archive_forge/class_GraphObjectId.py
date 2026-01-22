from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
class GraphObjectId(str):
    """
    An unique ID for a graph object (AudioContext, AudioNode, AudioParam) in Web Audio API
    """

    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> GraphObjectId:
        return cls(json)

    def __repr__(self):
        return 'GraphObjectId({})'.format(super().__repr__())