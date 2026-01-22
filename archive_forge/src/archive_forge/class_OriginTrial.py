from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
@dataclass
class OriginTrial:
    trial_name: str
    status: OriginTrialStatus
    tokens_with_status: typing.List[OriginTrialTokenWithStatus]

    def to_json(self):
        json = dict()
        json['trialName'] = self.trial_name
        json['status'] = self.status.to_json()
        json['tokensWithStatus'] = [i.to_json() for i in self.tokens_with_status]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(trial_name=str(json['trialName']), status=OriginTrialStatus.from_json(json['status']), tokens_with_status=[OriginTrialTokenWithStatus.from_json(i) for i in json['tokensWithStatus']])