from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import emulation
from . import io
from . import page
from . import runtime
from . import security
@dataclass
class AuthChallenge:
    """
    Authorization challenge for HTTP status code 401 or 407.
    """
    origin: str
    scheme: str
    realm: str
    source: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['origin'] = self.origin
        json['scheme'] = self.scheme
        json['realm'] = self.realm
        if self.source is not None:
            json['source'] = self.source
        return json

    @classmethod
    def from_json(cls, json):
        return cls(origin=str(json['origin']), scheme=str(json['scheme']), realm=str(json['realm']), source=str(json['source']) if 'source' in json else None)