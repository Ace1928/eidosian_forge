from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import io
class StreamFormat(enum.Enum):
    """
    Data format of a trace. Can be either the legacy JSON format or the
    protocol buffer format. Note that the JSON format will be deprecated soon.
    """
    JSON = 'json'
    PROTO = 'proto'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)