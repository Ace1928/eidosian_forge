import inspect
from collections import OrderedDict
from typing import List
from ray.dag import ClassNode, DAGNode
from ray.dag.function_node import FunctionNode
from ray.dag.utils import _DAGNodeNameGenerator
from ray.experimental.gradio_utils import type_to_string
from ray.serve._private.constants import (
from ray.serve._private.deployment_executor_node import DeploymentExecutorNode
from ray.serve._private.deployment_function_executor_node import (
from ray.serve._private.deployment_function_node import DeploymentFunctionNode
from ray.serve._private.deployment_node import DeploymentNode
from ray.serve.deployment import Deployment, schema_to_deployment
from ray.serve.handle import DeploymentHandle, RayServeHandle
from ray.serve.schema import DeploymentSchema
def transform_serve_dag_to_serve_executor_dag(serve_dag_root_node: DAGNode):
    """Given a runnable serve dag with deployment init args and options
    processed, transform into an equivalent, but minimal dag optimized for
    execution.
    """
    if isinstance(serve_dag_root_node, DeploymentNode):
        return DeploymentExecutorNode(serve_dag_root_node._deployment_handle, serve_dag_root_node.get_args(), serve_dag_root_node.get_kwargs())
    elif isinstance(serve_dag_root_node, DeploymentFunctionNode):
        return DeploymentFunctionExecutorNode(serve_dag_root_node._deployment_handle, serve_dag_root_node.get_args(), serve_dag_root_node.get_kwargs(), other_args_to_resolve=serve_dag_root_node.get_other_args_to_resolve())
    else:
        return serve_dag_root_node