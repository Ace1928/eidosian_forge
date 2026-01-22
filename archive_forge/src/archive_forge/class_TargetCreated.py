from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import page
@event_class('Target.targetCreated')
@dataclass
class TargetCreated:
    """
    Issued when a possible inspection target is created.
    """
    target_info: TargetInfo

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> TargetCreated:
        return cls(target_info=TargetInfo.from_json(json['targetInfo']))