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
class CachedResource:
    """
    Information about the cached resource.
    """
    url: str
    type_: ResourceType
    body_size: float
    response: typing.Optional[Response] = None

    def to_json(self):
        json = dict()
        json['url'] = self.url
        json['type'] = self.type_.to_json()
        json['bodySize'] = self.body_size
        if self.response is not None:
            json['response'] = self.response.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(url=str(json['url']), type_=ResourceType.from_json(json['type']), body_size=float(json['bodySize']), response=Response.from_json(json['response']) if 'response' in json else None)