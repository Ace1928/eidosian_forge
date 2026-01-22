import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import requests
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.util import NodeID, NodeIP, NodeKind, NodeStatus, NodeType
from ray.autoscaler.batching_node_provider import (
from ray.autoscaler.tags import (
def _worker_group_max_replicas(raycluster: Dict[str, Any], group_index: int) -> Optional[int]:
    """Extract the maxReplicas of a worker group.

    If maxReplicas is unset, return None, to be interpreted as "no constraint".
    At time of writing, it should be impossible for maxReplicas to be unset, but it's
    better to handle this anyway.
    """
    return raycluster['spec']['workerGroupSpecs'][group_index].get('maxReplicas')