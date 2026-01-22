from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
@dataclass
class CreditCard:
    number: str
    name: str
    expiry_month: str
    expiry_year: str
    cvc: str

    def to_json(self):
        json = dict()
        json['number'] = self.number
        json['name'] = self.name
        json['expiryMonth'] = self.expiry_month
        json['expiryYear'] = self.expiry_year
        json['cvc'] = self.cvc
        return json

    @classmethod
    def from_json(cls, json):
        return cls(number=str(json['number']), name=str(json['name']), expiry_month=str(json['expiryMonth']), expiry_year=str(json['expiryYear']), cvc=str(json['cvc']))