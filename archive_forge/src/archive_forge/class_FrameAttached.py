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
@event_class('Page.frameAttached')
@dataclass
class FrameAttached:
    """
    Fired when frame has been attached to its parent.
    """
    frame_id: FrameId
    parent_frame_id: FrameId
    stack: typing.Optional[runtime.StackTrace]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FrameAttached:
        return cls(frame_id=FrameId.from_json(json['frameId']), parent_frame_id=FrameId.from_json(json['parentFrameId']), stack=runtime.StackTrace.from_json(json['stack']) if 'stack' in json else None)