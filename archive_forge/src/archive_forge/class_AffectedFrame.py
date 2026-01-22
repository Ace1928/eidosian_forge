from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
from . import page
@dataclass
class AffectedFrame:
    """
    Information about the frame affected by an inspector issue.
    """
    frame_id: page.FrameId

    def to_json(self):
        json = dict()
        json['frameId'] = self.frame_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(frame_id=page.FrameId.from_json(json['frameId']))