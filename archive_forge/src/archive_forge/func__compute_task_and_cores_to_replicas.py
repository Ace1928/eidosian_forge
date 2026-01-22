import enum
import math
from typing import List, Optional, Tuple
import numpy as np
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu.topology import Topology
from tensorflow.python.util.tf_export import tf_export
def _compute_task_and_cores_to_replicas(core_assignment, topology):
    """Computes a nested dict which maps task and logical core to replicas."""
    task_and_cores_to_replicas = {}
    for replica in range(core_assignment.shape[0]):
        for logical_core in range(core_assignment.shape[1]):
            coordinates = core_assignment[replica, logical_core, :]
            task_id = topology.task_ordinal_at_coordinates(coordinates)
            if task_id not in task_and_cores_to_replicas:
                task_and_cores_to_replicas[task_id] = {}
            if logical_core not in task_and_cores_to_replicas[task_id]:
                task_and_cores_to_replicas[task_id][logical_core] = set()
            task_and_cores_to_replicas[task_id][logical_core].add(replica)
    task_to_sorted_replica_id = {}
    for task, core_to_replicas in task_and_cores_to_replicas.items():
        core_to_sorted_replicas = {}
        for core, replicas in core_to_replicas.items():
            core_to_sorted_replicas[core] = sorted(replicas)
        task_to_sorted_replica_id[task] = core_to_sorted_replicas
    return task_to_sorted_replica_id