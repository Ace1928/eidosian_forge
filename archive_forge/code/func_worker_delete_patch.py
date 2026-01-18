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
def worker_delete_patch(group_index: str, workers_to_delete: List[NodeID]):
    path = f'/spec/workerGroupSpecs/{group_index}/scaleStrategy'
    value = {'workersToDelete': workers_to_delete}
    return replace_patch(path, value)