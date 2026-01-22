from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
@dataclass
class BackForwardCacheNotRestoredExplanationTree:
    url: str
    explanations: typing.List[BackForwardCacheNotRestoredExplanation]
    children: typing.List[BackForwardCacheNotRestoredExplanationTree]

    def to_json(self):
        json = dict()
        json['url'] = self.url
        json['explanations'] = [i.to_json() for i in self.explanations]
        json['children'] = [i.to_json() for i in self.children]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(url=str(json['url']), explanations=[BackForwardCacheNotRestoredExplanation.from_json(i) for i in json['explanations']], children=[BackForwardCacheNotRestoredExplanationTree.from_json(i) for i in json['children']])