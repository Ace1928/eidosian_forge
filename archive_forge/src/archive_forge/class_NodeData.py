import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.util import NodeID, NodeIP, NodeKind, NodeStatus, NodeType
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
@dataclass
class NodeData:
    """Stores all data about a Ray node needed by the autoscaler.

    Attributes:
        kind: Whether the node is the head or a worker.
        type: The user-defined type of the node.
        ip: Cluster-internal ip of the node. ip can be None if the ip
            has not yet been assigned.
        status: The status of the node. You must adhere to the following semantics
            for status:
            * The status must be "up-to-date" if and only if the node is running.
            * The status must be "update-failed" if and only if the node is in an
                unknown or failed state.
            * If the node is in a pending (starting-up) state, the status should be
                a brief user-facing description of why the node is pending.
    """
    kind: NodeKind
    type: NodeType
    ip: Optional[NodeIP]
    status: NodeStatus