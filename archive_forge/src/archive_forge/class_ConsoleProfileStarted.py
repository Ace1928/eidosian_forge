from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import runtime
@event_class('Profiler.consoleProfileStarted')
@dataclass
class ConsoleProfileStarted:
    """
    Sent when new profile recording is started using console.profile() call.
    """
    id_: str
    location: debugger.Location
    title: typing.Optional[str]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ConsoleProfileStarted:
        return cls(id_=str(json['id']), location=debugger.Location.from_json(json['location']), title=str(json['title']) if 'title' in json else None)