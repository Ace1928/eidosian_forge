import time
from dataclasses import dataclass
import logging
from typing import List, Tuple, Any, Dict, Callable, TYPE_CHECKING
import ray
from ray import ObjectRef
from ray._private import signature
from ray.dag import DAGNode
from ray.workflow import workflow_context
from ray.workflow.workflow_context import get_task_status_info
from ray.workflow import serialization_context
from ray.workflow import workflow_storage
from ray.workflow.common import (
from ray.workflow.workflow_state import WorkflowExecutionState
from ray.workflow.workflow_state_from_dag import workflow_state_from_dag
def _workflow_task_executor(func: Callable, context: 'WorkflowTaskContext', task_id: 'TaskID', baked_inputs: '_BakedWorkflowInputs', runtime_options: 'WorkflowTaskRuntimeOptions') -> Tuple[Any, Any]:
    """Executor function for workflow task.

    Args:
        task_id: ID of the task.
        func: The workflow task function.
        baked_inputs: The processed inputs for the task.
        context: Workflow task context. Used to access correct storage etc.
        runtime_options: Parameters for workflow task execution.

    Returns:
        Workflow task output.
    """
    with workflow_context.workflow_task_context(context):
        store = workflow_storage.get_workflow_storage()
        args, kwargs = baked_inputs.resolve(store)
        try:
            store.save_task_prerun_metadata(task_id, {'start_time': time.time()})
            with workflow_context.workflow_execution():
                logger.info(f'{get_task_status_info(WorkflowStatus.RUNNING)}')
                output = func(*args, **kwargs)
            store.save_task_postrun_metadata(task_id, {'end_time': time.time()})
        except Exception as e:
            store.save_task_output(task_id, None, exception=e)
            raise e
        if isinstance(output, DAGNode):
            output = workflow_state_from_dag(output, None, context.workflow_id)
            execution_metadata = WorkflowExecutionMetadata(is_output_workflow=True)
        else:
            execution_metadata = WorkflowExecutionMetadata()
            if runtime_options.catch_exceptions:
                output = (output, None)
        if CheckpointMode(runtime_options.checkpoint) == CheckpointMode.SYNC:
            if isinstance(output, WorkflowExecutionState):
                store.save_workflow_execution_state(task_id, output)
            else:
                store.save_task_output(task_id, output, exception=None)
        return (execution_metadata, output)