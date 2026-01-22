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
class ContentSecurityPolicyIssueDetails:
    violated_directive: str
    is_report_only: bool
    content_security_policy_violation_type: ContentSecurityPolicyViolationType
    blocked_url: typing.Optional[str] = None
    frame_ancestor: typing.Optional[AffectedFrame] = None
    source_code_location: typing.Optional[SourceCodeLocation] = None
    violating_node_id: typing.Optional[dom.BackendNodeId] = None

    def to_json(self):
        json = dict()
        json['violatedDirective'] = self.violated_directive
        json['isReportOnly'] = self.is_report_only
        json['contentSecurityPolicyViolationType'] = self.content_security_policy_violation_type.to_json()
        if self.blocked_url is not None:
            json['blockedURL'] = self.blocked_url
        if self.frame_ancestor is not None:
            json['frameAncestor'] = self.frame_ancestor.to_json()
        if self.source_code_location is not None:
            json['sourceCodeLocation'] = self.source_code_location.to_json()
        if self.violating_node_id is not None:
            json['violatingNodeId'] = self.violating_node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(violated_directive=str(json['violatedDirective']), is_report_only=bool(json['isReportOnly']), content_security_policy_violation_type=ContentSecurityPolicyViolationType.from_json(json['contentSecurityPolicyViolationType']), blocked_url=str(json['blockedURL']) if 'blockedURL' in json else None, frame_ancestor=AffectedFrame.from_json(json['frameAncestor']) if 'frameAncestor' in json else None, source_code_location=SourceCodeLocation.from_json(json['sourceCodeLocation']) if 'sourceCodeLocation' in json else None, violating_node_id=dom.BackendNodeId.from_json(json['violatingNodeId']) if 'violatingNodeId' in json else None)