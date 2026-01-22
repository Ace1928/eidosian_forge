from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import page
@event_class('Target.attachedToTarget')
@dataclass
class AttachedToTarget:
    """
    **EXPERIMENTAL**

    Issued when attached to target because of auto-attach or ``attachToTarget`` command.
    """
    session_id: SessionID
    target_info: TargetInfo
    waiting_for_debugger: bool

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AttachedToTarget:
        return cls(session_id=SessionID.from_json(json['sessionId']), target_info=TargetInfo.from_json(json['targetInfo']), waiting_for_debugger=bool(json['waitingForDebugger']))