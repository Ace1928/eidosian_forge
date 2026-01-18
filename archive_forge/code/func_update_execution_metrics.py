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
def update_execution_metrics(self, dataset_tag: str, op_metrics: List[OpRuntimeMetrics], operator_tags: List[str], state: Dict[str, Any], force_update: bool=False):
    op_metrics_dicts = [metric.as_dict() for metric in op_metrics]
    args = (dataset_tag, op_metrics_dicts, operator_tags, state)
    if force_update:
        self._stats_actor().update_execution_metrics.remote(*args)
    else:
        with self._stats_lock:
            self._last_execution_stats[dataset_tag] = args
        self._start_thread_if_not_running()