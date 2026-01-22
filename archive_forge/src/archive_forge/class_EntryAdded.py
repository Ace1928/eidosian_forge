from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
from . import runtime
@event_class('Log.entryAdded')
@dataclass
class EntryAdded:
    """
    Issued when new message was logged.
    """
    entry: LogEntry

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> EntryAdded:
        return cls(entry=LogEntry.from_json(json['entry']))