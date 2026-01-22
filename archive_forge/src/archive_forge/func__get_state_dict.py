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
def _get_state_dict(self, state):
    last_op, last_state = list(self._topology.items())[-1]
    return {'state': state, 'progress': last_state.num_completed_tasks, 'total': last_op.num_outputs_total(), 'end_time': time.time() if state != 'RUNNING' else None, 'operators': {f'{op.name}{i}': {'progress': op_state.num_completed_tasks, 'total': op.num_outputs_total(), 'state': state} for i, (op, op_state) in enumerate(self._topology.items())}}