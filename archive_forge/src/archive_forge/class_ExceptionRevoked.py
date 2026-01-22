from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('Runtime.exceptionRevoked')
@dataclass
class ExceptionRevoked:
    """
    Issued when unhandled exception was revoked.
    """
    reason: str
    exception_id: int

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ExceptionRevoked:
        return cls(reason=str(json['reason']), exception_id=int(json['exceptionId']))