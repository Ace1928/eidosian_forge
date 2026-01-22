from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import runtime
@dataclass
class ScriptCoverage:
    """
    Coverage data for a JavaScript script.
    """
    script_id: runtime.ScriptId
    url: str
    functions: typing.List[FunctionCoverage]

    def to_json(self):
        json = dict()
        json['scriptId'] = self.script_id.to_json()
        json['url'] = self.url
        json['functions'] = [i.to_json() for i in self.functions]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(script_id=runtime.ScriptId.from_json(json['scriptId']), url=str(json['url']), functions=[FunctionCoverage.from_json(i) for i in json['functions']])