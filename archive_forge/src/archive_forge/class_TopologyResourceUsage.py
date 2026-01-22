import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import ray
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.autoscaling_requester import (
from ray.data._internal.execution.backpressure_policy import BackpressurePolicy
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.interfaces.physical_operator import (
from ray.data._internal.execution.operators.base_physical_operator import (
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.util import memory_string
from ray.data._internal.progress_bar import ProgressBar
from ray.data.context import DataContext
@dataclass
class TopologyResourceUsage:
    """Snapshot of resource usage in a `Topology` object.

    The stats here can be computed on the fly from any `Topology`; this class
    serves only a convenience wrapper to access the current usage snapshot.
    """
    overall: ExecutionResources
    downstream_memory_usage: Dict[PhysicalOperator, 'DownstreamMemoryInfo']

    @staticmethod
    def of(topology: Topology) -> 'TopologyResourceUsage':
        """Calculate the resource usage of the given topology."""
        downstream_usage = {}
        cur_usage = ExecutionResources(0, 0, 0)
        for op, state in reversed(topology.items()):
            cur_usage = cur_usage.add(op.current_resource_usage())
            if not isinstance(op, InputDataBuffer):
                cur_usage.object_store_memory += state.outqueue_memory_usage()
            f = (1.0 + len(downstream_usage)) / max(1.0, len(topology) - 1.0)
            downstream_usage[op] = DownstreamMemoryInfo(topology_fraction=min(1.0, f), object_store_memory=cur_usage.object_store_memory)
        return TopologyResourceUsage(cur_usage, downstream_usage)