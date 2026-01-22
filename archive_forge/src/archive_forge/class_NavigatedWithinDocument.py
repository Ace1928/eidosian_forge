from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
@event_class('Page.navigatedWithinDocument')
@dataclass
class NavigatedWithinDocument:
    """
    **EXPERIMENTAL**

    Fired when same-document navigation happens, e.g. due to history API usage or anchor navigation.
    """
    frame_id: FrameId
    url: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> NavigatedWithinDocument:
        return cls(frame_id=FrameId.from_json(json['frameId']), url=str(json['url']))