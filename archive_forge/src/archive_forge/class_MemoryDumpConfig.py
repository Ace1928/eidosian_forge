from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import io
class MemoryDumpConfig(dict):
    """
    Configuration for memory dump. Used only when "memory-infra" category is enabled.
    """

    def to_json(self) -> dict:
        return self

    @classmethod
    def from_json(cls, json: dict) -> MemoryDumpConfig:
        return cls(json)

    def __repr__(self):
        return 'MemoryDumpConfig({})'.format(super().__repr__())