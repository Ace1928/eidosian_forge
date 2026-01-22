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
class CrossOriginOpenerPolicyStatus:
    value: CrossOriginOpenerPolicyValue
    report_only_value: CrossOriginOpenerPolicyValue
    reporting_endpoint: typing.Optional[str] = None
    report_only_reporting_endpoint: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['value'] = self.value.to_json()
        json['reportOnlyValue'] = self.report_only_value.to_json()
        if self.reporting_endpoint is not None:
            json['reportingEndpoint'] = self.reporting_endpoint
        if self.report_only_reporting_endpoint is not None:
            json['reportOnlyReportingEndpoint'] = self.report_only_reporting_endpoint
        return json

    @classmethod
    def from_json(cls, json):
        return cls(value=CrossOriginOpenerPolicyValue.from_json(json['value']), report_only_value=CrossOriginOpenerPolicyValue.from_json(json['reportOnlyValue']), reporting_endpoint=str(json['reportingEndpoint']) if 'reportingEndpoint' in json else None, report_only_reporting_endpoint=str(json['reportOnlyReportingEndpoint']) if 'reportOnlyReportingEndpoint' in json else None)