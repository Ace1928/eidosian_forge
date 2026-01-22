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
@dataclass
class PermissionsPolicyBlockLocator:
    frame_id: FrameId
    block_reason: PermissionsPolicyBlockReason

    def to_json(self):
        json = dict()
        json['frameId'] = self.frame_id.to_json()
        json['blockReason'] = self.block_reason.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(frame_id=FrameId.from_json(json['frameId']), block_reason=PermissionsPolicyBlockReason.from_json(json['blockReason']))