from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
@dataclass
class MediaQuery:
    """
    Media query descriptor.
    """
    expressions: typing.List[MediaQueryExpression]
    active: bool

    def to_json(self):
        json = dict()
        json['expressions'] = [i.to_json() for i in self.expressions]
        json['active'] = self.active
        return json

    @classmethod
    def from_json(cls, json):
        return cls(expressions=[MediaQueryExpression.from_json(i) for i in json['expressions']], active=bool(json['active']))