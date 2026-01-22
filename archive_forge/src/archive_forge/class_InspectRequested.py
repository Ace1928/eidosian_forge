from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('Runtime.inspectRequested')
@dataclass
class InspectRequested:
    """
    Issued when object should be inspected (for example, as a result of inspect() command line API
    call).
    """
    object_: RemoteObject
    hints: dict

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> InspectRequested:
        return cls(object_=RemoteObject.from_json(json['object']), hints=dict(json['hints']))