from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
@dataclass
class AttributionReportingAggregatableTriggerData:
    key_piece: UnsignedInt128AsBase16
    source_keys: typing.List[str]
    filters: AttributionReportingFilterPair

    def to_json(self):
        json = dict()
        json['keyPiece'] = self.key_piece.to_json()
        json['sourceKeys'] = [i for i in self.source_keys]
        json['filters'] = self.filters.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(key_piece=UnsignedInt128AsBase16.from_json(json['keyPiece']), source_keys=[str(i) for i in json['sourceKeys']], filters=AttributionReportingFilterPair.from_json(json['filters']))