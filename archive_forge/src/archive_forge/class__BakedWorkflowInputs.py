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
@dataclass
class _BakedWorkflowInputs:
    """This class stores pre-processed inputs for workflow task execution.
    Especially, all input workflows to the workflow task will be scheduled,
    and their outputs (ObjectRefs) replace the original workflows."""
    args: 'ObjectRef'
    workflow_refs: 'List[WorkflowRef]'

    def resolve(self, store: workflow_storage.WorkflowStorage) -> Tuple[List, Dict]:
        """
        This function resolves the inputs for the code inside
        a workflow task (works on the callee side). For outputs from other
        workflows, we resolve them into object instances inplace.

        For each ObjectRef argument, the function returns both the ObjectRef
        and the object instance. If the ObjectRef is a chain of nested
        ObjectRefs, then we resolve it recursively until we get the
        object instance, and we return the *direct* ObjectRef of the
        instance. This function does not resolve ObjectRef
        inside another object (e.g. list of ObjectRefs) to give users some
        flexibility.

        Returns:
            Instances of arguments.
        """
        workflow_ref_mapping = []
        for r in self.workflow_refs:
            if r.ref is None:
                workflow_ref_mapping.append(store.load_task_output(r.task_id))
            else:
                workflow_ref_mapping.append(r.ref)
        with serialization_context.workflow_args_resolving_context(workflow_ref_mapping):
            flattened_args: List[Any] = ray.get(self.args)
        flattened_args = [ray.get(a) if isinstance(a, ObjectRef) else a for a in flattened_args]
        return signature.recover_args(flattened_args)