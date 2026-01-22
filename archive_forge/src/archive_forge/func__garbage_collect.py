from typing import Dict, List, Iterator, Optional, Tuple, TYPE_CHECKING
import asyncio
import logging
import time
from collections import defaultdict
import ray
from ray.exceptions import RayTaskError, RayError
from ray.workflow.common import (
from ray.workflow.exceptions import WorkflowCancellationError, WorkflowExecutionError
from ray.workflow.task_executor import get_task_executor, _BakedWorkflowInputs
from ray.workflow.workflow_state import (
def _garbage_collect(self) -> None:
    """Garbage collect the output refs of tasks.

        Currently, this is done after task submission, because when a task
        starts, we no longer needs its inputs (i.e. outputs from other tasks).

        # TODO(suquark): We may need to improve garbage collection
        #  when taking more fault tolerant cases into consideration.
        """
    state = self._state
    while state.free_outputs:
        gc_task_id = state.free_outputs.pop()
        assert state.get_input(gc_task_id) is not None
        state.output_map.pop(gc_task_id, None)