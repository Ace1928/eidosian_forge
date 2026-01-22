from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
@event_class('DOM.childNodeCountUpdated')
@dataclass
class ChildNodeCountUpdated:
    """
    Fired when ``Container``'s child node count has changed.
    """
    node_id: NodeId
    child_node_count: int

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ChildNodeCountUpdated:
        return cls(node_id=NodeId.from_json(json['nodeId']), child_node_count=int(json['childNodeCount']))