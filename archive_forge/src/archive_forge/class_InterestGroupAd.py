from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
@dataclass
class InterestGroupAd:
    """
    Ad advertising element inside an interest group.
    """
    render_url: str
    metadata: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['renderURL'] = self.render_url
        if self.metadata is not None:
            json['metadata'] = self.metadata
        return json

    @classmethod
    def from_json(cls, json):
        return cls(render_url=str(json['renderURL']), metadata=str(json['metadata']) if 'metadata' in json else None)