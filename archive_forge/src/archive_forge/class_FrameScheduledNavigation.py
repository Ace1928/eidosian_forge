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
@event_class('Page.frameScheduledNavigation')
@dataclass
class FrameScheduledNavigation:
    """
    Fired when frame schedules a potential navigation.
    """
    frame_id: FrameId
    delay: float
    reason: ClientNavigationReason
    url: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> FrameScheduledNavigation:
        return cls(frame_id=FrameId.from_json(json['frameId']), delay=float(json['delay']), reason=ClientNavigationReason.from_json(json['reason']), url=str(json['url']))