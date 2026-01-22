from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
from . import runtime
@dataclass
class ViolationSetting:
    """
    Violation configuration setting.
    """
    name: str
    threshold: float

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['threshold'] = self.threshold
        return json

    @classmethod
    def from_json(cls, json):
        return cls(name=str(json['name']), threshold=float(json['threshold']))