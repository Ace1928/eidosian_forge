from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
from . import page
@dataclass
class MixedContentIssueDetails:
    resolution_status: MixedContentResolutionStatus
    insecure_url: str
    main_resource_url: str
    resource_type: typing.Optional[MixedContentResourceType] = None
    request: typing.Optional[AffectedRequest] = None
    frame: typing.Optional[AffectedFrame] = None

    def to_json(self):
        json = dict()
        json['resolutionStatus'] = self.resolution_status.to_json()
        json['insecureURL'] = self.insecure_url
        json['mainResourceURL'] = self.main_resource_url
        if self.resource_type is not None:
            json['resourceType'] = self.resource_type.to_json()
        if self.request is not None:
            json['request'] = self.request.to_json()
        if self.frame is not None:
            json['frame'] = self.frame.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(resolution_status=MixedContentResolutionStatus.from_json(json['resolutionStatus']), insecure_url=str(json['insecureURL']), main_resource_url=str(json['mainResourceURL']), resource_type=MixedContentResourceType.from_json(json['resourceType']) if 'resourceType' in json else None, request=AffectedRequest.from_json(json['request']) if 'request' in json else None, frame=AffectedFrame.from_json(json['frame']) if 'frame' in json else None)