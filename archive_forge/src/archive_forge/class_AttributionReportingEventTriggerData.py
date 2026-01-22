from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
@dataclass
class AttributionReportingEventTriggerData:
    data: UnsignedInt64AsBase10
    priority: SignedInt64AsBase10
    filters: AttributionReportingFilterPair
    dedup_key: typing.Optional[UnsignedInt64AsBase10] = None

    def to_json(self):
        json = dict()
        json['data'] = self.data.to_json()
        json['priority'] = self.priority.to_json()
        json['filters'] = self.filters.to_json()
        if self.dedup_key is not None:
            json['dedupKey'] = self.dedup_key.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(data=UnsignedInt64AsBase10.from_json(json['data']), priority=SignedInt64AsBase10.from_json(json['priority']), filters=AttributionReportingFilterPair.from_json(json['filters']), dedup_key=UnsignedInt64AsBase10.from_json(json['dedupKey']) if 'dedupKey' in json else None)