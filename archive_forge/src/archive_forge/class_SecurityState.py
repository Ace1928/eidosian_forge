from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
class SecurityState(enum.Enum):
    """
    The security level of a page or resource.
    """
    UNKNOWN = 'unknown'
    NEUTRAL = 'neutral'
    INSECURE = 'insecure'
    SECURE = 'secure'
    INFO = 'info'
    INSECURE_BROKEN = 'insecure-broken'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)