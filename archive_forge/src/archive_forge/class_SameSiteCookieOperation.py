from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
from . import page
class SameSiteCookieOperation(enum.Enum):
    SET_COOKIE = 'SetCookie'
    READ_COOKIE = 'ReadCookie'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)