from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import runtime
@dataclass
class ScriptTypeProfile:
    """
    Type profile data collected during runtime for a JavaScript script.
    """
    script_id: runtime.ScriptId
    url: str
    entries: typing.List[TypeProfileEntry]

    def to_json(self):
        json = dict()
        json['scriptId'] = self.script_id.to_json()
        json['url'] = self.url
        json['entries'] = [i.to_json() for i in self.entries]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(script_id=runtime.ScriptId.from_json(json['scriptId']), url=str(json['url']), entries=[TypeProfileEntry.from_json(i) for i in json['entries']])