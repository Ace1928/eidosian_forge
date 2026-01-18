from typing import Optional
from collections import deque
from ray.workflow import serialization
from ray.workflow.common import TaskID, WorkflowRef
from ray.workflow.exceptions import WorkflowTaskNotRecoverableError
from ray.workflow import workflow_storage
from ray.workflow.workflow_state import WorkflowExecutionState, Task
Try to construct a workflow (task) that recovers the workflow task.
    If the workflow task already has an output checkpointing file, we return
    the workflow task id instead.

    Args:
        workflow_id: The ID of the workflow.
        task_id: The ID of the output task. If None, it will be the entrypoint of
            the workflow.

    Returns:
        A workflow that recovers the task, or the output of the task
            if it has been checkpointed.
    