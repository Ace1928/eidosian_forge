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
class FrameResourceTree:
    """
    Information about the Frame hierarchy along with their cached resources.
    """
    frame: Frame
    resources: typing.List[FrameResource]
    child_frames: typing.Optional[typing.List[FrameResourceTree]] = None

    def to_json(self):
        json = dict()
        json['frame'] = self.frame.to_json()
        json['resources'] = [i.to_json() for i in self.resources]
        if self.child_frames is not None:
            json['childFrames'] = [i.to_json() for i in self.child_frames]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(frame=Frame.from_json(json['frame']), resources=[FrameResource.from_json(i) for i in json['resources']], child_frames=[FrameResourceTree.from_json(i) for i in json['childFrames']] if 'childFrames' in json else None)