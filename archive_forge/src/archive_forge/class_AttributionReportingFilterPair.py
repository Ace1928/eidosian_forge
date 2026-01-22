from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
@dataclass
class AttributionReportingFilterPair:
    filters: typing.List[AttributionReportingFilterConfig]
    not_filters: typing.List[AttributionReportingFilterConfig]

    def to_json(self):
        json = dict()
        json['filters'] = [i.to_json() for i in self.filters]
        json['notFilters'] = [i.to_json() for i in self.not_filters]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(filters=[AttributionReportingFilterConfig.from_json(i) for i in json['filters']], not_filters=[AttributionReportingFilterConfig.from_json(i) for i in json['notFilters']])