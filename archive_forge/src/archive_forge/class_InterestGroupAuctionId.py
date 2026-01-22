from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
class InterestGroupAuctionId(str):
    """
    Protected audience interest group auction identifier.
    """

    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> InterestGroupAuctionId:
        return cls(json)

    def __repr__(self):
        return 'InterestGroupAuctionId({})'.format(super().__repr__())