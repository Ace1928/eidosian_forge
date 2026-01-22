from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
@event_class('Storage.interestGroupAccessed')
@dataclass
class InterestGroupAccessed:
    """
    One of the interest groups was accessed. Note that these events are global
    to all targets sharing an interest group store.
    """
    access_time: network.TimeSinceEpoch
    type_: InterestGroupAccessType
    owner_origin: str
    name: str
    component_seller_origin: typing.Optional[str]
    bid: typing.Optional[float]
    bid_currency: typing.Optional[str]
    unique_auction_id: typing.Optional[InterestGroupAuctionId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> InterestGroupAccessed:
        return cls(access_time=network.TimeSinceEpoch.from_json(json['accessTime']), type_=InterestGroupAccessType.from_json(json['type']), owner_origin=str(json['ownerOrigin']), name=str(json['name']), component_seller_origin=str(json['componentSellerOrigin']) if 'componentSellerOrigin' in json else None, bid=float(json['bid']) if 'bid' in json else None, bid_currency=str(json['bidCurrency']) if 'bidCurrency' in json else None, unique_auction_id=InterestGroupAuctionId.from_json(json['uniqueAuctionId']) if 'uniqueAuctionId' in json else None)