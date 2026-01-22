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
class CookiePriority(enum.Enum):
    """
    Represents the cookie's 'Priority' status:
    https://tools.ietf.org/html/draft-west-cookie-priority-00
    """
    LOW = 'Low'
    MEDIUM = 'Medium'
    HIGH = 'High'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)