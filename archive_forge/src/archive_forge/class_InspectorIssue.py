from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
from . import page
@dataclass
class InspectorIssue:
    """
    An inspector issue reported from the back-end.
    """
    code: InspectorIssueCode
    details: InspectorIssueDetails

    def to_json(self):
        json = dict()
        json['code'] = self.code.to_json()
        json['details'] = self.details.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(code=InspectorIssueCode.from_json(json['code']), details=InspectorIssueDetails.from_json(json['details']))