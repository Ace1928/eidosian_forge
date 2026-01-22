from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
class AttributionReportingSourceRegistrationTimeConfig(enum.Enum):
    INCLUDE = 'include'
    EXCLUDE = 'exclude'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)