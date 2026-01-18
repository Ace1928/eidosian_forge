import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import ray
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.autoscaling_requester import (
from ray.data._internal.execution.backpressure_policy import BackpressurePolicy
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.interfaces.physical_operator import (
from ray.data._internal.execution.operators.base_physical_operator import (
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.util import memory_string
from ray.data._internal.progress_bar import ProgressBar
from ray.data.context import DataContext
def update_operator_states(topology: Topology) -> None:
    """Update operator states accordingly for newly completed tasks.
    Should be called after `process_completed_tasks()`."""
    for op, op_state in topology.items():
        if op_state.inputs_done_called:
            continue
        all_inputs_done = True
        for idx, dep in enumerate(op.input_dependencies):
            if dep.completed() and (not topology[dep].outqueue):
                if not op_state.input_done_called[idx]:
                    op.input_done(idx)
                    op_state.input_done_called[idx] = True
            else:
                all_inputs_done = False
        if all_inputs_done:
            op.all_inputs_done()
            op_state.inputs_done_called = True
    for op, op_state in reversed(list(topology.items())):
        if op_state.dependents_completed_called:
            continue
        dependents_completed = len(op.output_dependencies) > 0 and all((not dep.need_more_inputs() for dep in op.output_dependencies))
        if dependents_completed:
            op.all_dependents_complete()
            op_state.dependents_completed_called = True