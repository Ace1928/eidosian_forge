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
@event_class('Network.loadingFinished')
@dataclass
class LoadingFinished:
    """
    Fired when HTTP request has finished loading.
    """
    request_id: RequestId
    timestamp: MonotonicTime
    encoded_data_length: float

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> LoadingFinished:
        return cls(request_id=RequestId.from_json(json['requestId']), timestamp=MonotonicTime.from_json(json['timestamp']), encoded_data_length=float(json['encodedDataLength']))