from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
class InterestGroupAuctionEventType(enum.Enum):
    """
    Enum of auction events.
    """
    STARTED = 'started'
    CONFIG_RESOLVED = 'configResolved'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)