import collections
import copy
import logging
import os
from abc import abstractmethod
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple
import ray
from ray._private.gcs_utils import PlacementGroupTableData
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.loader import load_function_or_class
from ray.autoscaler._private.node_provider_availability_tracker import (
from ray.autoscaler._private.util import (
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
from ray.core.generated.common_pb2 import PlacementStrategy
def _node_type_counts_to_node_resources(node_types: Dict[NodeType, NodeTypeConfigDict], node_type_counts: Dict[NodeType, int]) -> List[ResourceDict]:
    """Converts a node_type_counts dict into a list of node_resources."""
    resources = []
    for node_type, count in node_type_counts.items():
        resources += [node_types[node_type]['resources'].copy() for _ in range(count)]
    return resources