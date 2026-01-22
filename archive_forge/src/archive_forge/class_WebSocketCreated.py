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
@event_class('Network.webSocketCreated')
@dataclass
class WebSocketCreated:
    """
    Fired upon WebSocket creation.
    """
    request_id: RequestId
    url: str
    initiator: typing.Optional[Initiator]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WebSocketCreated:
        return cls(request_id=RequestId.from_json(json['requestId']), url=str(json['url']), initiator=Initiator.from_json(json['initiator']) if 'initiator' in json else None)