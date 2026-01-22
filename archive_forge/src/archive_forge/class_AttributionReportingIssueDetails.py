from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
from . import runtime
@dataclass
class AttributionReportingIssueDetails:
    """
    Details for issues around "Attribution Reporting API" usage.
    Explainer: https://github.com/WICG/attribution-reporting-api
    """
    violation_type: AttributionReportingIssueType
    request: typing.Optional[AffectedRequest] = None
    violating_node_id: typing.Optional[dom.BackendNodeId] = None
    invalid_parameter: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['violationType'] = self.violation_type.to_json()
        if self.request is not None:
            json['request'] = self.request.to_json()
        if self.violating_node_id is not None:
            json['violatingNodeId'] = self.violating_node_id.to_json()
        if self.invalid_parameter is not None:
            json['invalidParameter'] = self.invalid_parameter
        return json

    @classmethod
    def from_json(cls, json):
        return cls(violation_type=AttributionReportingIssueType.from_json(json['violationType']), request=AffectedRequest.from_json(json['request']) if 'request' in json else None, violating_node_id=dom.BackendNodeId.from_json(json['violatingNodeId']) if 'violatingNodeId' in json else None, invalid_parameter=str(json['invalidParameter']) if 'invalidParameter' in json else None)