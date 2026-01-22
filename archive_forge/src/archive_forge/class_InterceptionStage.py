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
class InterceptionStage(enum.Enum):
    """
    Stages of the interception to begin intercepting. Request will intercept before the request is
    sent. Response will intercept after the response is received.
    """
    REQUEST = 'Request'
    HEADERS_RECEIVED = 'HeadersReceived'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)