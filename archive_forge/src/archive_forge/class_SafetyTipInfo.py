from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
@dataclass
class SafetyTipInfo:
    safety_tip_status: SafetyTipStatus
    safe_url: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['safetyTipStatus'] = self.safety_tip_status.to_json()
        if self.safe_url is not None:
            json['safeUrl'] = self.safe_url
        return json

    @classmethod
    def from_json(cls, json):
        return cls(safety_tip_status=SafetyTipStatus.from_json(json['safetyTipStatus']), safe_url=str(json['safeUrl']) if 'safeUrl' in json else None)