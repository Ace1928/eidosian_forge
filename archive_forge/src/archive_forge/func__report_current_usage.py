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
def _report_current_usage(self, cur_usage: TopologyResourceUsage, limits: ExecutionResources) -> None:
    resources_status = f'Running: {cur_usage.overall.cpu}/{limits.cpu} CPU, {cur_usage.overall.gpu}/{limits.gpu} GPU, {cur_usage.overall.object_store_memory_str()}/{limits.object_store_memory_str()} object_store_memory'
    if self._global_info:
        self._global_info.set_description(resources_status)