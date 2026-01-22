from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('Runtime.executionContextDestroyed')
@dataclass
class ExecutionContextDestroyed:
    """
    Issued when execution context is destroyed.
    """
    execution_context_id: ExecutionContextId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ExecutionContextDestroyed:
        return cls(execution_context_id=ExecutionContextId.from_json(json['executionContextId']))