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
For each strict spread, attempt to reserve as much space as possible
        on the node, then allocate new nodes for the unfulfilled portion.

        Args:
            strict_spreads (List[List[ResourceDict]]): A list of placement
                groups which must be spread out.
            node_resources (List[ResourceDict]): Available node resources in
                the cluster.
            node_type_counts (Dict[NodeType, int]): The amount of each type of
                node pending or in the cluster.
            utilization_scorer: A function that, given a node
                type, its resources, and resource demands, returns what its
                utilization would be.

        Returns:
            Dict[NodeType, int]: Nodes to add.
            List[ResourceDict]: The updated node_resources after the method.
            Dict[NodeType, int]: The updated node_type_counts.

        