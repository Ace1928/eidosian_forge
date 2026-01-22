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
def _log_op_metrics(topology: Topology) -> None:
    """Logs the metrics of each operator.

    Args:
        topology: The topology to debug.
    """
    log_str = 'Operator Metrics:\n'
    for op in topology:
        log_str += f'{op.name}: {op.metrics.as_dict()}\n'
    logger.get_logger(log_to_stdout=False).info(log_str)