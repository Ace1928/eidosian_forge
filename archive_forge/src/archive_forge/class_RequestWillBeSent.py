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
@event_class('Network.requestWillBeSent')
@dataclass
class RequestWillBeSent:
    """
    Fired when page is about to send HTTP request.
    """
    request_id: RequestId
    loader_id: LoaderId
    document_url: str
    request: Request
    timestamp: MonotonicTime
    wall_time: TimeSinceEpoch
    initiator: Initiator
    redirect_has_extra_info: bool
    redirect_response: typing.Optional[Response]
    type_: typing.Optional[ResourceType]
    frame_id: typing.Optional[page.FrameId]
    has_user_gesture: typing.Optional[bool]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> RequestWillBeSent:
        return cls(request_id=RequestId.from_json(json['requestId']), loader_id=LoaderId.from_json(json['loaderId']), document_url=str(json['documentURL']), request=Request.from_json(json['request']), timestamp=MonotonicTime.from_json(json['timestamp']), wall_time=TimeSinceEpoch.from_json(json['wallTime']), initiator=Initiator.from_json(json['initiator']), redirect_has_extra_info=bool(json['redirectHasExtraInfo']), redirect_response=Response.from_json(json['redirectResponse']) if 'redirectResponse' in json else None, type_=ResourceType.from_json(json['type']) if 'type' in json else None, frame_id=page.FrameId.from_json(json['frameId']) if 'frameId' in json else None, has_user_gesture=bool(json['hasUserGesture']) if 'hasUserGesture' in json else None)