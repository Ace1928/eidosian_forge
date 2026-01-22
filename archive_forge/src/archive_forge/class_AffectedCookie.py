from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
from . import page
@dataclass
class AffectedCookie:
    """
    Information about a cookie that is affected by an inspector issue.
    """
    name: str
    path: str
    domain: str

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['path'] = self.path
        json['domain'] = self.domain
        return json

    @classmethod
    def from_json(cls, json):
        return cls(name=str(json['name']), path=str(json['path']), domain=str(json['domain']))