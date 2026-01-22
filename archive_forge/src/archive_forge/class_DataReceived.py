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
@event_class('Network.dataReceived')
@dataclass
class DataReceived:
    """
    Fired when data chunk was received over the network.
    """
    request_id: RequestId
    timestamp: MonotonicTime
    data_length: int
    encoded_data_length: int
    data: typing.Optional[str]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DataReceived:
        return cls(request_id=RequestId.from_json(json['requestId']), timestamp=MonotonicTime.from_json(json['timestamp']), data_length=int(json['dataLength']), encoded_data_length=int(json['encodedDataLength']), data=str(json['data']) if 'data' in json else None)