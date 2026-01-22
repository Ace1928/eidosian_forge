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
@event_class('Network.subresourceWebBundleMetadataReceived')
@dataclass
class SubresourceWebBundleMetadataReceived:
    """
    **EXPERIMENTAL**

    Fired once when parsing the .wbn file has succeeded.
    The event contains the information about the web bundle contents.
    """
    request_id: RequestId
    urls: typing.List[str]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> SubresourceWebBundleMetadataReceived:
        return cls(request_id=RequestId.from_json(json['requestId']), urls=[str(i) for i in json['urls']])