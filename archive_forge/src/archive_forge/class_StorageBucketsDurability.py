from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
class StorageBucketsDurability(enum.Enum):
    RELAXED = 'relaxed'
    STRICT = 'strict'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)