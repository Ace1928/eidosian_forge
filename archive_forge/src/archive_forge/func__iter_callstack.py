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
def _iter_callstack(self, task_id: TaskID) -> Iterator[Tuple[TaskID, Task]]:
    state = self._state
    while task_id in state.task_context and task_id in state.tasks:
        yield (task_id, state.tasks[task_id])
        task_id = state.task_context[task_id].creator_task_id