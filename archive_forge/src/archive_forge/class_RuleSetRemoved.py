from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
@event_class('Preload.ruleSetRemoved')
@dataclass
class RuleSetRemoved:
    id_: RuleSetId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> RuleSetRemoved:
        return cls(id_=RuleSetId.from_json(json['id']))