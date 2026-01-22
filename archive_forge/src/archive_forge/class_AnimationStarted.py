from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime
@event_class('Animation.animationStarted')
@dataclass
class AnimationStarted:
    """
    Event for animation that has been started.
    """
    animation: Animation

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AnimationStarted:
        return cls(animation=Animation.from_json(json['animation']))