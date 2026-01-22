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
class Initiator:
    """
    Information about the request initiator.
    """
    type_: str
    stack: typing.Optional[runtime.StackTrace] = None
    url: typing.Optional[str] = None
    line_number: typing.Optional[float] = None
    column_number: typing.Optional[float] = None
    request_id: typing.Optional[RequestId] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        if self.stack is not None:
            json['stack'] = self.stack.to_json()
        if self.url is not None:
            json['url'] = self.url
        if self.line_number is not None:
            json['lineNumber'] = self.line_number
        if self.column_number is not None:
            json['columnNumber'] = self.column_number
        if self.request_id is not None:
            json['requestId'] = self.request_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(type_=str(json['type']), stack=runtime.StackTrace.from_json(json['stack']) if 'stack' in json else None, url=str(json['url']) if 'url' in json else None, line_number=float(json['lineNumber']) if 'lineNumber' in json else None, column_number=float(json['columnNumber']) if 'columnNumber' in json else None, request_id=RequestId.from_json(json['requestId']) if 'requestId' in json else None)