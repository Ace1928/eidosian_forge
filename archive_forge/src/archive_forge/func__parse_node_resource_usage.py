from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from itertools import chain
from typing import Any, Dict, List, Tuple
import ray
from ray._private.ray_constants import AUTOSCALER_NAMESPACE, AUTOSCALER_V2_ENABLED_KEY
from ray._private.utils import binary_to_hex
from ray.autoscaler._private.autoscaler import AutoscalerSummary
from ray.autoscaler._private.node_provider_availability_tracker import (
from ray.autoscaler._private.util import LoadMetricsSummary, format_info_string
from ray.autoscaler.v2.schema import (
from ray.core.generated.autoscaler_pb2 import (
from ray.core.generated.autoscaler_pb2 import (
from ray.core.generated.instance_manager_pb2 import Instance
from ray.experimental.internal_kv import _internal_kv_get, _internal_kv_initialized
@classmethod
def _parse_node_resource_usage(cls, node_state: NodeState, usage: Dict[str, ResourceUsage]) -> Dict[str, ResourceUsage]:
    """
        Parse the node resource usage from the node state.
        Args:
            node_state: the node state
            usage: the usage dict to be updated. This is a dict of
                {resource_name: ResourceUsage}
        Returns:
            usage: the updated usage dict
        """
    d = defaultdict(lambda: [0.0, 0.0])
    for resource_name, resource_total in node_state.total_resources.items():
        d[resource_name][1] += resource_total
        d[resource_name][0] += resource_total
    for resource_name, resource_available in node_state.available_resources.items():
        d[resource_name][0] -= resource_available
    for k, (used, total) in d.items():
        usage[k].resource_name = k
        usage[k].used += used
        usage[k].total += total
    return usage