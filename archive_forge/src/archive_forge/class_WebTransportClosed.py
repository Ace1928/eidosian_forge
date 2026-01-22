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
@event_class('Network.webTransportClosed')
@dataclass
class WebTransportClosed:
    """
    Fired when WebTransport is disposed.
    """
    transport_id: RequestId
    timestamp: MonotonicTime

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WebTransportClosed:
        return cls(transport_id=RequestId.from_json(json['transportId']), timestamp=MonotonicTime.from_json(json['timestamp']))