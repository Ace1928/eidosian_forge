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
def _create_placement_group(self):
    """Creates a placement group if it does not exist.

        If a placement group is already detected (Tune) this will be a no-op.

        By default the placement group will be created with PACK strategy.
        This is optimized for colocating GPUs on a minimal number of nodes.
        This behavior can be overridden to use the SPREAD strategy by defining
        ``TRAIN_ENABLE_WORKER_SPREAD_ENV``

        If a placement group is created it will be stored as
        self._placement_group.
        """
    current_placement_group = get_current_placement_group()
    worker = ray._private.worker.global_worker
    should_capture_child_tasks_in_placement_group = worker.should_capture_child_tasks_in_placement_group
    should_create_placement_group = current_placement_group is None or not should_capture_child_tasks_in_placement_group
    if should_create_placement_group:
        additional_resources_per_worker = self._additional_resources_per_worker or {}
        bundle = {'CPU': self._num_cpus_per_worker, 'GPU': self._num_gpus_per_worker, **additional_resources_per_worker}
        bundles = [bundle.copy() for _ in range(self._num_workers)]
        use_spread = bool(env_integer(TRAIN_ENABLE_WORKER_SPREAD_ENV, 0))
        strategy = 'SPREAD' if use_spread else 'PACK'
        placement_group = ray.util.placement_group(bundles, strategy=strategy)
        logger.debug('Waiting for placement group to start.')
        timeout = env_integer(TRAIN_PLACEMENT_GROUP_TIMEOUT_S_ENV, 100)
        ready, _ = ray.wait([placement_group.ready()], timeout=timeout)
        if ready:
            logger.debug('Placement group has started.')
        else:
            raise TimeoutError('Placement group creation timed out. Make sure your cluster either has enough resources or use an autoscaling cluster. If you are running on a cluster, make sure you specify an address in `ray.init()`, for example, `ray.init("auto")`. You can also increase the timeout by setting the TRAIN_PLACEMENT_GROUP_TIMEOUT_S environment variable. Current resources available: {}, resources requested by the placement group: {}'.format(ray.available_resources(), placement_group.bundle_specs))
        self._placement_group = placement_group