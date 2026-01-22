from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import dom_debugger
from . import page
@dataclass
class RareBooleanData:
    index: typing.List[int]

    def to_json(self):
        json = dict()
        json['index'] = [i for i in self.index]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(index=[int(i) for i in json['index']])