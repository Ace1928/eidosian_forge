from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@dataclass
class CachedResponse:
    """
    Cached response
    """
    body: str

    def to_json(self):
        json = dict()
        json['body'] = self.body
        return json

    @classmethod
    def from_json(cls, json):
        return cls(body=str(json['body']))