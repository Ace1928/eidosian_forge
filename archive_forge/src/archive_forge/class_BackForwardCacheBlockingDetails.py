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
class BackForwardCacheBlockingDetails:
    line_number: int
    column_number: int
    url: typing.Optional[str] = None
    function: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['lineNumber'] = self.line_number
        json['columnNumber'] = self.column_number
        if self.url is not None:
            json['url'] = self.url
        if self.function is not None:
            json['function'] = self.function
        return json

    @classmethod
    def from_json(cls, json):
        return cls(line_number=int(json['lineNumber']), column_number=int(json['columnNumber']), url=str(json['url']) if 'url' in json else None, function=str(json['function']) if 'function' in json else None)