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
def _execution_allowed(op: PhysicalOperator, global_usage: TopologyResourceUsage, global_limits: ExecutionResources) -> bool:
    """Return whether an operator is allowed to execute given resource usage.

    Operators are throttled globally based on CPU and GPU limits for the stream.

    For an N operator DAG, we only throttle the kth operator (in the source-to-sink
    ordering) on object store utilization if the cumulative object store utilization
    for the kth operator and every operator downstream from it is greater than
    k/N * global_limit; i.e., the N - k operator sub-DAG is using more object store
    memory than it's share.

    Args:
        op: The operator to check.
        global_usage: Resource usage across the entire topology.
        global_limits: Execution resource limits.

    Returns:
        Whether the op is allowed to run.
    """
    if op.throttling_disabled():
        return True
    assert isinstance(global_usage, TopologyResourceUsage), global_usage
    global_floored = ExecutionResources(cpu=math.floor(global_usage.overall.cpu or 0), gpu=math.floor(global_usage.overall.gpu or 0), object_store_memory=global_usage.overall.object_store_memory)
    inc = op.incremental_resource_usage()
    if inc.cpu and inc.gpu:
        raise NotImplementedError('Operator incremental resource usage cannot specify both CPU and GPU at the same time, since it may cause deadlock.')
    inc_indicator = ExecutionResources(cpu=1 if inc.cpu else 0, gpu=1 if inc.gpu else 0, object_store_memory=inc.object_store_memory if DataContext.get_current().use_runtime_metrics_scheduling else None)
    new_usage = global_floored.add(inc_indicator)
    if new_usage.satisfies_limit(global_limits):
        return True
    global_limits_sans_memory = ExecutionResources(cpu=global_limits.cpu, gpu=global_limits.gpu)
    global_ok_sans_memory = new_usage.satisfies_limit(global_limits_sans_memory)
    downstream_usage = global_usage.downstream_memory_usage[op]
    downstream_limit = global_limits.scale(downstream_usage.topology_fraction)
    downstream_memory_ok = ExecutionResources(object_store_memory=downstream_usage.object_store_memory).satisfies_limit(downstream_limit)
    if DataContext.get_current().use_runtime_metrics_scheduling and global_ok_sans_memory and (op.metrics.average_bytes_change_per_task is not None) and (op.metrics.average_bytes_change_per_task <= 0):
        return True
    return global_ok_sans_memory and downstream_memory_ok