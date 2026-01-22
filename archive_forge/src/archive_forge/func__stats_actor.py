import collections
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4
import numpy as np
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces.op_runtime_metrics import OpRuntimeMetrics
from ray.data._internal.util import capfirst
from ray.data.block import BlockMetadata
from ray.data.context import DataContext
from ray.util.annotations import DeveloperAPI
from ray.util.metrics import Gauge
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def _stats_actor(self, create_if_not_exists=True) -> _StatsActor:
    if ray._private.worker._global_node is None:
        raise RuntimeError('Global node is not initialized.')
    current_cluster_id = ray._private.worker._global_node.cluster_id
    if self._stats_actor_handle is None or self._stats_actor_cluster_id != current_cluster_id:
        if create_if_not_exists:
            self._stats_actor_handle = _get_or_create_stats_actor()
        else:
            self._stat_actor_handle = ray.get_actor(name=STATS_ACTOR_NAME, namespace=STATS_ACTOR_NAMESPACE)
        self._stats_actor_cluster_id = current_cluster_id
    return self._stats_actor_handle