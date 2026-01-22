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
@event_class('Network.requestIntercepted')
@dataclass
class RequestIntercepted:
    """
    **EXPERIMENTAL**

    Details of an intercepted HTTP request, which must be either allowed, blocked, modified or
    mocked.
    Deprecated, use Fetch.requestPaused instead.
    """
    interception_id: InterceptionId
    request: Request
    frame_id: page.FrameId
    resource_type: ResourceType
    is_navigation_request: bool
    is_download: typing.Optional[bool]
    redirect_url: typing.Optional[str]
    auth_challenge: typing.Optional[AuthChallenge]
    response_error_reason: typing.Optional[ErrorReason]
    response_status_code: typing.Optional[int]
    response_headers: typing.Optional[Headers]
    request_id: typing.Optional[RequestId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> RequestIntercepted:
        return cls(interception_id=InterceptionId.from_json(json['interceptionId']), request=Request.from_json(json['request']), frame_id=page.FrameId.from_json(json['frameId']), resource_type=ResourceType.from_json(json['resourceType']), is_navigation_request=bool(json['isNavigationRequest']), is_download=bool(json['isDownload']) if 'isDownload' in json else None, redirect_url=str(json['redirectUrl']) if 'redirectUrl' in json else None, auth_challenge=AuthChallenge.from_json(json['authChallenge']) if 'authChallenge' in json else None, response_error_reason=ErrorReason.from_json(json['responseErrorReason']) if 'responseErrorReason' in json else None, response_status_code=int(json['responseStatusCode']) if 'responseStatusCode' in json else None, response_headers=Headers.from_json(json['responseHeaders']) if 'responseHeaders' in json else None, request_id=RequestId.from_json(json['requestId']) if 'requestId' in json else None)