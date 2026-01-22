from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
@event_class('HeapProfiler.lastSeenObjectId')
@dataclass
class LastSeenObjectId:
    """
    If heap objects tracking has been started then backend regularly sends a current value for last
    seen object id and corresponding timestamp. If the were changes in the heap since last event
    then one or more heapStatsUpdate events will be sent before a new lastSeenObjectId event.
    """
    last_seen_object_id: int
    timestamp: float

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> LastSeenObjectId:
        return cls(last_seen_object_id=int(json['lastSeenObjectId']), timestamp=float(json['timestamp']))