from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
@dataclass
class TrustTokens:
    """
    Pair of issuer origin and number of available (signed, but not used) Trust
    Tokens from that issuer.
    """
    issuer_origin: str
    count: float

    def to_json(self):
        json = dict()
        json['issuerOrigin'] = self.issuer_origin
        json['count'] = self.count
        return json

    @classmethod
    def from_json(cls, json):
        return cls(issuer_origin=str(json['issuerOrigin']), count=float(json['count']))