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
@event_class('Page.frameStoppedLoading')
@dataclass
class FrameStoppedLoading:
    """
    **EXPERIMENTAL**

    Fired when frame has stopped loading.
    """
    frame_id: FrameId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FrameStoppedLoading:
        return cls(frame_id=FrameId.from_json(json['frameId']))