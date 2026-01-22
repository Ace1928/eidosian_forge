import ray
from ray.dag.dag_node import DAGNode
from ray.dag.input_node import InputNode
from ray.dag.format_utils import get_dag_node_str
from ray.dag.constants import (
from ray.util.annotations import DeveloperAPI
from typing import Any, Dict, List, Optional, Tuple
@DeveloperAPI
class ClassMethodNode(DAGNode):
    """Represents an actor method invocation in a Ray function DAG."""

    def __init__(self, method_name: str, method_args: Tuple[Any], method_kwargs: Dict[str, Any], method_options: Dict[str, Any], other_args_to_resolve: Dict[str, Any]):
        self._bound_args = method_args or []
        self._bound_kwargs = method_kwargs or {}
        self._bound_options = method_options or {}
        self._method_name: str = method_name
        self._parent_class_node: ClassNode = other_args_to_resolve.get(PARENT_CLASS_NODE_KEY)
        self._prev_class_method_call: Optional[ClassMethodNode] = other_args_to_resolve.get(PREV_CLASS_METHOD_CALL_KEY, None)
        super().__init__(method_args, method_kwargs, method_options, other_args_to_resolve=other_args_to_resolve)

    def _copy_impl(self, new_args: List[Any], new_kwargs: Dict[str, Any], new_options: Dict[str, Any], new_other_args_to_resolve: Dict[str, Any]):
        return ClassMethodNode(self._method_name, new_args, new_kwargs, new_options, other_args_to_resolve=new_other_args_to_resolve)

    def _execute_impl(self, *args, **kwargs):
        """Executor of ClassMethodNode by ray.remote()

        Args and kwargs are to match base class signature, but not in the
        implementation. All args and kwargs should be resolved and replaced
        with value in bound_args and bound_kwargs via bottom-up recursion when
        current node is executed.
        """
        method_body = getattr(self._parent_class_node, self._method_name)
        return method_body.options(**self._bound_options).remote(*self._bound_args, **self._bound_kwargs)

    def __str__(self) -> str:
        return get_dag_node_str(self, f'{self._method_name}()')

    def get_method_name(self) -> str:
        return self._method_name