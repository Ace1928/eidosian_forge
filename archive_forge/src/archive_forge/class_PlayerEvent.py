from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@dataclass
class PlayerEvent:
    """
    Corresponds to kMediaEventTriggered
    """
    timestamp: Timestamp
    value: str

    def to_json(self):
        json = dict()
        json['timestamp'] = self.timestamp.to_json()
        json['value'] = self.value
        return json

    @classmethod
    def from_json(cls, json):
        return cls(timestamp=Timestamp.from_json(json['timestamp']), value=str(json['value']))