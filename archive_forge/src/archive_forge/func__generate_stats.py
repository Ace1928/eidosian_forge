import os
import threading
import time
import uuid
from typing import Dict, Iterator, List, Optional
import ray
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.autoscaling_requester import (
from ray.data._internal.execution.backpressure_policy import (
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.streaming_executor_state import (
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.stats import DatasetStats, StatsManager
from ray.data.context import DataContext
def _generate_stats(self) -> DatasetStats:
    """Create a new stats object reflecting execution status so far."""
    stats = self._initial_stats or DatasetStats(stages={}, parent=None)
    for op in self._topology:
        if isinstance(op, InputDataBuffer):
            continue
        builder = stats.child_builder(op.name, override_start_time=self._start_time)
        stats = builder.build_multistage(op.get_stats())
        stats.extra_metrics = op.metrics.as_dict()
    return stats