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
@event_class('Network.webTransportCreated')
@dataclass
class WebTransportCreated:
    """
    Fired upon WebTransport creation.
    """
    transport_id: RequestId
    url: str
    timestamp: MonotonicTime
    initiator: typing.Optional[Initiator]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WebTransportCreated:
        return cls(transport_id=RequestId.from_json(json['transportId']), url=str(json['url']), timestamp=MonotonicTime.from_json(json['timestamp']), initiator=Initiator.from_json(json['initiator']) if 'initiator' in json else None)