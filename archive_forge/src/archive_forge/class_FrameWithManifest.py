from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
@dataclass
class FrameWithManifest:
    """
    Frame identifier - manifest URL pair.
    """
    frame_id: page.FrameId
    manifest_url: str
    status: int

    def to_json(self):
        json = dict()
        json['frameId'] = self.frame_id.to_json()
        json['manifestURL'] = self.manifest_url
        json['status'] = self.status
        return json

    @classmethod
    def from_json(cls, json):
        return cls(frame_id=page.FrameId.from_json(json['frameId']), manifest_url=str(json['manifestURL']), status=int(json['status']))