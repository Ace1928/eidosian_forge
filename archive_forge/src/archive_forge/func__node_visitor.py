from typing import Any, List, Optional
import re
import unicodedata
import ray
from ray.workflow.common import WORKFLOW_OPTIONS
from ray.dag import DAGNode, FunctionNode, InputNode
from ray.dag.input_node import InputAttributeNode, DAGInputData
from ray import cloudpickle
from ray._private import signature
from ray._private.client_mode_hook import client_mode_should_convert
from ray.workflow import serialization_context
from ray.workflow.common import (
from ray.workflow import workflow_context
from ray.workflow.workflow_state import WorkflowExecutionState, Task
def _node_visitor(node: Any) -> Any:
    if isinstance(node, FunctionNode):
        bound_options = node._bound_options.copy()
        num_returns = bound_options.get('num_returns', 1)
        if num_returns is None:
            num_returns = 1
        if num_returns > 1:
            raise ValueError('Workflow task can only have one return.')
        workflow_options = bound_options.get('_metadata', {}).get(WORKFLOW_OPTIONS, {})
        checkpoint = workflow_options.get('checkpoint', None)
        if checkpoint is None:
            checkpoint = context.checkpoint if context is not None else True
        catch_exceptions = workflow_options.get('catch_exceptions', None)
        if catch_exceptions is None:
            if node.get_stable_uuid() == dag_node.get_stable_uuid():
                catch_exceptions = context.catch_exceptions if context is not None else False
            else:
                catch_exceptions = False
        max_retries = bound_options.get('max_retries', 3)
        retry_exceptions = bound_options.get('retry_exceptions', False)
        task_options = WorkflowTaskRuntimeOptions(task_type=TaskType.FUNCTION, catch_exceptions=catch_exceptions, retry_exceptions=retry_exceptions, max_retries=max_retries, checkpoint=checkpoint, ray_options=bound_options)
        workflow_refs: List[WorkflowRef] = []
        with serialization_context.workflow_args_serialization_context(workflow_refs):
            _func_signature = signature.extract_signature(node._body)
            flattened_args = signature.flatten_args(_func_signature, node._bound_args, node._bound_kwargs)
            if client_mode_should_convert():
                flattened_args = _SerializationContextPreservingWrapper(flattened_args)
            input_placeholder: ray.ObjectRef = ray.put(flattened_args, _owner=mgr)
        orig_task_id = workflow_options.get('task_id', None)
        if orig_task_id is None:
            orig_task_id = f'{get_module(node._body)}.{slugify(get_qualname(node._body))}'
        task_id = ray.get(mgr.gen_task_id.remote(workflow_id, orig_task_id))
        state.add_dependencies(task_id, [s.task_id for s in workflow_refs])
        state.task_input_args[task_id] = input_placeholder
        user_metadata = workflow_options.get('metadata', {})
        validate_user_metadata(user_metadata)
        state.tasks[task_id] = Task(task_id=task_id, options=task_options, user_metadata=user_metadata, func_body=node._body)
        return WorkflowRef(task_id)
    if isinstance(node, InputAttributeNode):
        return node._execute_impl()
    if isinstance(node, InputNode):
        return input_context
    if not isinstance(node, DAGNode):
        return node
    raise TypeError(f'Unsupported DAG node: {node}')