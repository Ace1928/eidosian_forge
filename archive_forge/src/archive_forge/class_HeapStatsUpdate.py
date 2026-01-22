from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
@event_class('HeapProfiler.heapStatsUpdate')
@dataclass
class HeapStatsUpdate:
    """
    If heap objects tracking has been started then backend may send update for one or more fragments
    """
    stats_update: typing.List[int]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> HeapStatsUpdate:
        return cls(stats_update=[int(i) for i in json['statsUpdate']])