from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import io
from . import network
from . import page
@event_class('Fetch.authRequired')
@dataclass
class AuthRequired:
    """
    Issued when the domain is enabled with handleAuthRequests set to true.
    The request is paused until client responds with continueWithAuth.
    """
    request_id: RequestId
    request: network.Request
    frame_id: page.FrameId
    resource_type: network.ResourceType
    auth_challenge: AuthChallenge

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AuthRequired:
        return cls(request_id=RequestId.from_json(json['requestId']), request=network.Request.from_json(json['request']), frame_id=page.FrameId.from_json(json['frameId']), resource_type=network.ResourceType.from_json(json['resourceType']), auth_challenge=AuthChallenge.from_json(json['authChallenge']))