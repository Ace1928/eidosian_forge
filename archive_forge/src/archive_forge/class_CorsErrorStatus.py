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
class CorsErrorStatus:
    cors_error: CorsError
    failed_parameter: str

    def to_json(self):
        json = dict()
        json['corsError'] = self.cors_error.to_json()
        json['failedParameter'] = self.failed_parameter
        return json

    @classmethod
    def from_json(cls, json):
        return cls(cors_error=CorsError.from_json(json['corsError']), failed_parameter=str(json['failedParameter']))