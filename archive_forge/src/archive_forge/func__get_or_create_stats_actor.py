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
def _get_or_create_stats_actor():
    ctx = DataContext.get_current()
    scheduling_strategy = ctx.scheduling_strategy
    if not ray.util.client.ray.is_connected():
        scheduling_strategy = NodeAffinitySchedulingStrategy(ray.get_runtime_context().get_node_id(), soft=False)
    with _stats_actor_lock:
        return _StatsActor.options(name=STATS_ACTOR_NAME, namespace=STATS_ACTOR_NAMESPACE, get_if_exists=True, lifetime='detached', scheduling_strategy=scheduling_strategy).remote()