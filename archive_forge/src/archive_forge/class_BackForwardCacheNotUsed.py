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
@event_class('Page.backForwardCacheNotUsed')
@dataclass
class BackForwardCacheNotUsed:
    """
    **EXPERIMENTAL**

    Fired for failed bfcache history navigations if BackForwardCache feature is enabled. Do
    not assume any ordering with the Page.frameNavigated event. This event is fired only for
    main-frame history navigation where the document changes (non-same-document navigations),
    when bfcache navigation fails.
    """
    loader_id: network.LoaderId
    frame_id: FrameId
    not_restored_explanations: typing.List[BackForwardCacheNotRestoredExplanation]
    not_restored_explanations_tree: typing.Optional[BackForwardCacheNotRestoredExplanationTree]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> BackForwardCacheNotUsed:
        return cls(loader_id=network.LoaderId.from_json(json['loaderId']), frame_id=FrameId.from_json(json['frameId']), not_restored_explanations=[BackForwardCacheNotRestoredExplanation.from_json(i) for i in json['notRestoredExplanations']], not_restored_explanations_tree=BackForwardCacheNotRestoredExplanationTree.from_json(json['notRestoredExplanationsTree']) if 'notRestoredExplanationsTree' in json else None)