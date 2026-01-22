from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
@dataclass
class VisibleSecurityState:
    """
    Security state information about the page.
    """
    security_state: SecurityState
    security_state_issue_ids: typing.List[str]
    certificate_security_state: typing.Optional[CertificateSecurityState] = None
    safety_tip_info: typing.Optional[SafetyTipInfo] = None

    def to_json(self):
        json = dict()
        json['securityState'] = self.security_state.to_json()
        json['securityStateIssueIds'] = [i for i in self.security_state_issue_ids]
        if self.certificate_security_state is not None:
            json['certificateSecurityState'] = self.certificate_security_state.to_json()
        if self.safety_tip_info is not None:
            json['safetyTipInfo'] = self.safety_tip_info.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(security_state=SecurityState.from_json(json['securityState']), security_state_issue_ids=[str(i) for i in json['securityStateIssueIds']], certificate_security_state=CertificateSecurityState.from_json(json['certificateSecurityState']) if 'certificateSecurityState' in json else None, safety_tip_info=SafetyTipInfo.from_json(json['safetyTipInfo']) if 'safetyTipInfo' in json else None)