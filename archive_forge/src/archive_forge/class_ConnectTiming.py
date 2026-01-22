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
@dataclass
class ConnectTiming:
    request_time: float

    def to_json(self):
        json = dict()
        json['requestTime'] = self.request_time
        return json

    @classmethod
    def from_json(cls, json):
        return cls(request_time=float(json['requestTime']))