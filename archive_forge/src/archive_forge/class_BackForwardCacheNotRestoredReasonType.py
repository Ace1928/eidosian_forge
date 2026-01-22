from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
class BackForwardCacheNotRestoredReasonType(enum.Enum):
    """
    Types of not restored reasons for back-forward cache.
    """
    SUPPORT_PENDING = 'SupportPending'
    PAGE_SUPPORT_NEEDED = 'PageSupportNeeded'
    CIRCUMSTANTIAL = 'Circumstantial'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)