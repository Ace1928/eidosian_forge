from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
@event_class('Debugger.resumed')
@dataclass
class Resumed:
    """
    Fired when the virtual machine resumed execution.
    """

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> Resumed:
        return cls()