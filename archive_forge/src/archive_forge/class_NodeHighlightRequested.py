from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
@event_class('Overlay.nodeHighlightRequested')
@dataclass
class NodeHighlightRequested:
    """
    Fired when the node should be highlighted. This happens after call to ``setInspectMode``.
    """
    node_id: dom.NodeId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> NodeHighlightRequested:
        return cls(node_id=dom.NodeId.from_json(json['nodeId']))