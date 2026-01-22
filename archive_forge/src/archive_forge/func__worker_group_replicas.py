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
def _worker_group_replicas(raycluster: Dict[str, Any], group_index: int):
    return raycluster['spec']['workerGroupSpecs'][group_index].get('replicas', 1)