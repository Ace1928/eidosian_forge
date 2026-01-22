from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import runtime
@event_class('Profiler.consoleProfileFinished')
@dataclass
class ConsoleProfileFinished:
    id_: str
    location: debugger.Location
    profile: Profile
    title: typing.Optional[str]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ConsoleProfileFinished:
        return cls(id_=str(json['id']), location=debugger.Location.from_json(json['location']), profile=Profile.from_json(json['profile']), title=str(json['title']) if 'title' in json else None)