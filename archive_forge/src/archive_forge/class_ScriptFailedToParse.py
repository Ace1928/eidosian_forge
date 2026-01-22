from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
@event_class('Debugger.scriptFailedToParse')
@dataclass
class ScriptFailedToParse:
    """
    Fired when virtual machine fails to parse the script.
    """
    script_id: runtime.ScriptId
    url: str
    start_line: int
    start_column: int
    end_line: int
    end_column: int
    execution_context_id: runtime.ExecutionContextId
    hash_: str
    execution_context_aux_data: typing.Optional[dict]
    source_map_url: typing.Optional[str]
    has_source_url: typing.Optional[bool]
    is_module: typing.Optional[bool]
    length: typing.Optional[int]
    stack_trace: typing.Optional[runtime.StackTrace]
    code_offset: typing.Optional[int]
    script_language: typing.Optional[debugger.ScriptLanguage]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ScriptFailedToParse:
        return cls(script_id=runtime.ScriptId.from_json(json['scriptId']), url=str(json['url']), start_line=int(json['startLine']), start_column=int(json['startColumn']), end_line=int(json['endLine']), end_column=int(json['endColumn']), execution_context_id=runtime.ExecutionContextId.from_json(json['executionContextId']), hash_=str(json['hash']), execution_context_aux_data=dict(json['executionContextAuxData']) if 'executionContextAuxData' in json else None, source_map_url=str(json['sourceMapURL']) if 'sourceMapURL' in json else None, has_source_url=bool(json['hasSourceURL']) if 'hasSourceURL' in json else None, is_module=bool(json['isModule']) if 'isModule' in json else None, length=int(json['length']) if 'length' in json else None, stack_trace=runtime.StackTrace.from_json(json['stackTrace']) if 'stackTrace' in json else None, code_offset=int(json['codeOffset']) if 'codeOffset' in json else None, script_language=debugger.ScriptLanguage.from_json(json['scriptLanguage']) if 'scriptLanguage' in json else None)