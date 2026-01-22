from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
@dataclass
class BackendNode:
    """
    Backend node with a friendly name.
    """
    node_type: int
    node_name: str
    backend_node_id: BackendNodeId

    def to_json(self):
        json = dict()
        json['nodeType'] = self.node_type
        json['nodeName'] = self.node_name
        json['backendNodeId'] = self.backend_node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(node_type=int(json['nodeType']), node_name=str(json['nodeName']), backend_node_id=BackendNodeId.from_json(json['backendNodeId']))