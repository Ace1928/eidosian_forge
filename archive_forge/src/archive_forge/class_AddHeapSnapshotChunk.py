from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
@event_class('HeapProfiler.addHeapSnapshotChunk')
@dataclass
class AddHeapSnapshotChunk:
    chunk: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AddHeapSnapshotChunk:
        return cls(chunk=str(json['chunk']))