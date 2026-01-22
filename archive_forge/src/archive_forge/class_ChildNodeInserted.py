from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
@event_class('DOM.childNodeInserted')
@dataclass
class ChildNodeInserted:
    """
    Mirrors ``DOMNodeInserted`` event.
    """
    parent_node_id: NodeId
    previous_node_id: NodeId
    node: Node

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ChildNodeInserted:
        return cls(parent_node_id=NodeId.from_json(json['parentNodeId']), previous_node_id=NodeId.from_json(json['previousNodeId']), node=Node.from_json(json['node']))