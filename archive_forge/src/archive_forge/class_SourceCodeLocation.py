from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
from . import runtime
@dataclass
class SourceCodeLocation:
    url: str
    line_number: int
    column_number: int
    script_id: typing.Optional[runtime.ScriptId] = None

    def to_json(self):
        json = dict()
        json['url'] = self.url
        json['lineNumber'] = self.line_number
        json['columnNumber'] = self.column_number
        if self.script_id is not None:
            json['scriptId'] = self.script_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(url=str(json['url']), line_number=int(json['lineNumber']), column_number=int(json['columnNumber']), script_id=runtime.ScriptId.from_json(json['scriptId']) if 'scriptId' in json else None)