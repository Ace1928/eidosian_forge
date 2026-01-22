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
@dataclass
class AppManifestError:
    """
    Error while paring app manifest.
    """
    message: str
    critical: int
    line: int
    column: int

    def to_json(self):
        json = dict()
        json['message'] = self.message
        json['critical'] = self.critical
        json['line'] = self.line
        json['column'] = self.column
        return json

    @classmethod
    def from_json(cls, json):
        return cls(message=str(json['message']), critical=int(json['critical']), line=int(json['line']), column=int(json['column']))