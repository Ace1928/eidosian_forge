from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
@dataclass
class UserAgentBrandVersion:
    """
    Used to specify User Agent Cient Hints to emulate. See https://wicg.github.io/ua-client-hints
    """
    brand: str
    version: str

    def to_json(self):
        json = dict()
        json['brand'] = self.brand
        json['version'] = self.version
        return json

    @classmethod
    def from_json(cls, json):
        return cls(brand=str(json['brand']), version=str(json['version']))