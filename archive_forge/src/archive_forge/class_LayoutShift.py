from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
@dataclass
class LayoutShift:
    """
    See https://wicg.github.io/layout-instability/#sec-layout-shift and layout_shift.idl
    """
    value: float
    had_recent_input: bool
    last_input_time: network.TimeSinceEpoch
    sources: typing.List[LayoutShiftAttribution]

    def to_json(self):
        json = dict()
        json['value'] = self.value
        json['hadRecentInput'] = self.had_recent_input
        json['lastInputTime'] = self.last_input_time.to_json()
        json['sources'] = [i.to_json() for i in self.sources]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(value=float(json['value']), had_recent_input=bool(json['hadRecentInput']), last_input_time=network.TimeSinceEpoch.from_json(json['lastInputTime']), sources=[LayoutShiftAttribution.from_json(i) for i in json['sources']])