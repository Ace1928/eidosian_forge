from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('Cast.sinksUpdated')
@dataclass
class SinksUpdated:
    """
    This is fired whenever the list of available sinks changes. A sink is a
    device or a software surface that you can cast to.
    """
    sinks: typing.List[Sink]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> SinksUpdated:
        return cls(sinks=[Sink.from_json(i) for i in json['sinks']])