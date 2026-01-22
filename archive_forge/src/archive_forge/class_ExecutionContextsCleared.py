from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('Runtime.executionContextsCleared')
@dataclass
class ExecutionContextsCleared:
    """
    Issued when all executionContexts were cleared in browser
    """

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ExecutionContextsCleared:
        return cls()