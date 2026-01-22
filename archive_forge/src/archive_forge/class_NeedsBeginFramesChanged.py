from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('HeadlessExperimental.needsBeginFramesChanged')
@dataclass
class NeedsBeginFramesChanged:
    """
    Issued when the target starts or stops needing BeginFrames.
    Deprecated. Issue beginFrame unconditionally instead and use result from
    beginFrame to detect whether the frames were suppressed.
    """
    needs_begin_frames: bool

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> NeedsBeginFramesChanged:
        return cls(needs_begin_frames=bool(json['needsBeginFrames']))