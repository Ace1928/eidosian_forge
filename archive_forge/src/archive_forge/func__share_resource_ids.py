import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar
import ray
import ray._private.ray_constants as ray_constants
from ray._private.ray_constants import env_integer
from ray.data import Dataset
from ray.exceptions import RayActorError
from ray.train import Checkpoint, DataConfig
from ray.train._internal.session import (
from ray.train._internal.storage import StorageContext
from ray.train._internal.utils import check_for_failure
from ray.train._internal.worker_group import WorkerGroup
from ray.train.backend import BackendConfig
from ray.train.constants import (
from ray.util.placement_group import get_current_placement_group, remove_placement_group
def _share_resource_ids(self, resource: str, env_var: str):
    """Sets the given env_var on all workers.

        For each worker, the cores/devices are visible to all the
        workers on that worker's node.This allows workers on the
        same node to communicate with one another.

        Example:

            Setup:
            - Node1:
                - Worker1: {0, 1}
                - Worker2: {2, 3}
            - Node2:
                - Worker3: {0, 1}

            NEURON_RT_VISIBLE_CORES/TPU_VISIBLE_CHIPS/...:
            - Worker1: "0,1,2,3"
            - Worker2: "0,1,2,3"
            - Worker2: "0,1"

        Args:
            resource: The name of the resource/accelerator.
            env_var: The name of the environment variable to set.
        """
    node_ids_and_resource_ids = [(w.metadata.node_id, w.metadata.resource_ids[resource]) for w in self.worker_group.workers]
    node_id_to_worker_id = defaultdict(set)
    node_id_to_resource_ids = defaultdict(set)
    for worker_id, (node_id, resource_ids) in enumerate(node_ids_and_resource_ids):
        node_id_to_worker_id[node_id].add(worker_id)
        node_id_to_resource_ids[node_id].update(resource_ids)
    futures = []
    for node_id, resource_ids in node_id_to_resource_ids.items():
        resource_ids = sorted(resource_ids)
        all_resource_ids = ','.join(resource_ids)

        def set_resource_ids():
            os.environ[env_var] = all_resource_ids
        for worker_id in node_id_to_worker_id[node_id]:
            futures.append(self.worker_group.execute_single_async(worker_id, set_resource_ids))
    ray.get(futures)