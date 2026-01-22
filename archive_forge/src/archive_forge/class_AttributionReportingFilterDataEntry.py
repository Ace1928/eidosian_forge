from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
@dataclass
class AttributionReportingFilterDataEntry:
    key: str
    values: typing.List[str]

    def to_json(self):
        json = dict()
        json['key'] = self.key
        json['values'] = [i for i in self.values]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(key=str(json['key']), values=[str(i) for i in json['values']])