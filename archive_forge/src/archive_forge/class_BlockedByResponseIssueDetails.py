from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
from . import page
@dataclass
class BlockedByResponseIssueDetails:
    """
    Details for a request that has been blocked with the BLOCKED_BY_RESPONSE
    code. Currently only used for COEP/COOP, but may be extended to include
    some CSP errors in the future.
    """
    request: AffectedRequest
    reason: BlockedByResponseReason
    frame: typing.Optional[AffectedFrame] = None

    def to_json(self):
        json = dict()
        json['request'] = self.request.to_json()
        json['reason'] = self.reason.to_json()
        if self.frame is not None:
            json['frame'] = self.frame.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(request=AffectedRequest.from_json(json['request']), reason=BlockedByResponseReason.from_json(json['reason']), frame=AffectedFrame.from_json(json['frame']) if 'frame' in json else None)