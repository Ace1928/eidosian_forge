import ray
from ray.dag.dag_node import DAGNode
from ray.dag.input_node import InputNode
from ray.dag.format_utils import get_dag_node_str
from ray.dag.constants import (
from ray.util.annotations import DeveloperAPI
from typing import Any, Dict, List, Optional, Tuple
@DeveloperAPI
class ClassNode(DAGNode):
    """Represents an actor creation in a Ray task DAG."""

    def __init__(self, cls, cls_args, cls_kwargs, cls_options, other_args_to_resolve=None):
        self._body = cls
        self._last_call: Optional['ClassMethodNode'] = None
        super().__init__(cls_args, cls_kwargs, cls_options, other_args_to_resolve=other_args_to_resolve)
        if self._contains_input_node():
            raise ValueError('InputNode handles user dynamic input the the DAG, and cannot be used as args, kwargs, or other_args_to_resolve in ClassNode constructor because it is not available at class construction or binding time.')

    def _copy_impl(self, new_args: List[Any], new_kwargs: Dict[str, Any], new_options: Dict[str, Any], new_other_args_to_resolve: Dict[str, Any]):
        return ClassNode(self._body, new_args, new_kwargs, new_options, other_args_to_resolve=new_other_args_to_resolve)

    def _execute_impl(self, *args, **kwargs):
        """Executor of ClassNode by ray.remote()

        Args and kwargs are to match base class signature, but not in the
        implementation. All args and kwargs should be resolved and replaced
        with value in bound_args and bound_kwargs via bottom-up recursion when
        current node is executed.
        """
        return ray.remote(self._body).options(**self._bound_options).remote(*self._bound_args, **self._bound_kwargs)

    def _contains_input_node(self) -> bool:
        """Check if InputNode is used in children DAGNodes with current node
        as the root.
        """
        children_dag_nodes = self._get_all_child_nodes()
        for child in children_dag_nodes:
            if isinstance(child, InputNode):
                return True
        return False

    def __getattr__(self, method_name: str):
        if method_name == 'bind' and 'bind' not in dir(self._body):
            raise AttributeError(f'.bind() cannot be used again on {type(self)} ')
        getattr(self._body, method_name)
        call_node = _UnboundClassMethodNode(self, method_name)
        return call_node

    def __str__(self) -> str:
        return get_dag_node_str(self, str(self._body))