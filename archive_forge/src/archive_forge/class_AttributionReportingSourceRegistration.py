from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
@dataclass
class AttributionReportingSourceRegistration:
    time: network.TimeSinceEpoch
    expiry: int
    trigger_specs: typing.List[AttributionReportingTriggerSpec]
    aggregatable_report_window: int
    type_: AttributionReportingSourceType
    source_origin: str
    reporting_origin: str
    destination_sites: typing.List[str]
    event_id: UnsignedInt64AsBase10
    priority: SignedInt64AsBase10
    filter_data: typing.List[AttributionReportingFilterDataEntry]
    aggregation_keys: typing.List[AttributionReportingAggregationKeysEntry]
    trigger_data_matching: AttributionReportingTriggerDataMatching
    debug_key: typing.Optional[UnsignedInt64AsBase10] = None

    def to_json(self):
        json = dict()
        json['time'] = self.time.to_json()
        json['expiry'] = self.expiry
        json['triggerSpecs'] = [i.to_json() for i in self.trigger_specs]
        json['aggregatableReportWindow'] = self.aggregatable_report_window
        json['type'] = self.type_.to_json()
        json['sourceOrigin'] = self.source_origin
        json['reportingOrigin'] = self.reporting_origin
        json['destinationSites'] = [i for i in self.destination_sites]
        json['eventId'] = self.event_id.to_json()
        json['priority'] = self.priority.to_json()
        json['filterData'] = [i.to_json() for i in self.filter_data]
        json['aggregationKeys'] = [i.to_json() for i in self.aggregation_keys]
        json['triggerDataMatching'] = self.trigger_data_matching.to_json()
        if self.debug_key is not None:
            json['debugKey'] = self.debug_key.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(time=network.TimeSinceEpoch.from_json(json['time']), expiry=int(json['expiry']), trigger_specs=[AttributionReportingTriggerSpec.from_json(i) for i in json['triggerSpecs']], aggregatable_report_window=int(json['aggregatableReportWindow']), type_=AttributionReportingSourceType.from_json(json['type']), source_origin=str(json['sourceOrigin']), reporting_origin=str(json['reportingOrigin']), destination_sites=[str(i) for i in json['destinationSites']], event_id=UnsignedInt64AsBase10.from_json(json['eventId']), priority=SignedInt64AsBase10.from_json(json['priority']), filter_data=[AttributionReportingFilterDataEntry.from_json(i) for i in json['filterData']], aggregation_keys=[AttributionReportingAggregationKeysEntry.from_json(i) for i in json['aggregationKeys']], trigger_data_matching=AttributionReportingTriggerDataMatching.from_json(json['triggerDataMatching']), debug_key=UnsignedInt64AsBase10.from_json(json['debugKey']) if 'debugKey' in json else None)