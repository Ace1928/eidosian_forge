from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('Runtime.bindingCalled')
@dataclass
class BindingCalled:
    """
    **EXPERIMENTAL**

    Notification is issued every time when binding is called.
    """
    name: str
    payload: str
    execution_context_id: ExecutionContextId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> BindingCalled:
        return cls(name=str(json['name']), payload=str(json['payload']), execution_context_id=ExecutionContextId.from_json(json['executionContextId']))