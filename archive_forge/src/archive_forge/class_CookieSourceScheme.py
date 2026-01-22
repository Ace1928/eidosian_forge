from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import emulation
from . import io
from . import page
from . import runtime
from . import security
class CookieSourceScheme(enum.Enum):
    """
    Represents the source scheme of the origin that originally set the cookie.
    A value of "Unset" allows protocol clients to emulate legacy cookie scope for the scheme.
    This is a temporary ability and it will be removed in the future.
    """
    UNSET = 'Unset'
    NON_SECURE = 'NonSecure'
    SECURE = 'Secure'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)