from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@dataclass
class SamplingProfile:
    """
    Array of heap profile samples.
    """
    samples: typing.List[SamplingProfileNode]
    modules: typing.List[Module]

    def to_json(self):
        json = dict()
        json['samples'] = [i.to_json() for i in self.samples]
        json['modules'] = [i.to_json() for i in self.modules]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(samples=[SamplingProfileNode.from_json(i) for i in json['samples']], modules=[Module.from_json(i) for i in json['modules']])