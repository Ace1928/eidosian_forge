from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
from . import service_worker
@dataclass
class EventMetadata:
    """
    A key-value pair for additional event information to pass along.
    """
    key: str
    value: str

    def to_json(self):
        json = dict()
        json['key'] = self.key
        json['value'] = self.value
        return json

    @classmethod
    def from_json(cls, json):
        return cls(key=str(json['key']), value=str(json['value']))