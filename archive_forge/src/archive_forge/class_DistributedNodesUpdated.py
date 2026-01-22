from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
@event_class('DOM.distributedNodesUpdated')
@dataclass
class DistributedNodesUpdated:
    """
    **EXPERIMENTAL**

    Called when distribution is changed.
    """
    insertion_point_id: NodeId
    distributed_nodes: typing.List[BackendNode]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DistributedNodesUpdated:
        return cls(insertion_point_id=NodeId.from_json(json['insertionPointId']), distributed_nodes=[BackendNode.from_json(i) for i in json['distributedNodes']])