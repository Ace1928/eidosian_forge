from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('Runtime.consoleAPICalled')
@dataclass
class ConsoleAPICalled:
    """
    Issued when console API was called.
    """
    type_: str
    args: typing.List[RemoteObject]
    execution_context_id: ExecutionContextId
    timestamp: Timestamp
    stack_trace: typing.Optional[StackTrace]
    context: typing.Optional[str]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ConsoleAPICalled:
        return cls(type_=str(json['type']), args=[RemoteObject.from_json(i) for i in json['args']], execution_context_id=ExecutionContextId.from_json(json['executionContextId']), timestamp=Timestamp.from_json(json['timestamp']), stack_trace=StackTrace.from_json(json['stackTrace']) if 'stackTrace' in json else None, context=str(json['context']) if 'context' in json else None)