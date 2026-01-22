import ray
from ray.dag.base import DAGNodeBase
from ray.dag.py_obj_scanner import _PyObjScanner
from ray.util.annotations import DeveloperAPI
from typing import (
import uuid
import asyncio
def _apply_and_replace_all_child_nodes(self, fn: 'Callable[[DAGNode], T]') -> 'DAGNode':
    """Apply and replace all immediate child nodes using a given function.

        This is a shallow replacement only. To recursively transform nodes in
        the DAG, use ``apply_recursive()``.

        Args:
            fn: Callable that will be applied once to each child of this node.

        Returns:
            New DAGNode after replacing all child nodes.
        """
    replace_table = {}
    scanner = _PyObjScanner()
    for node in scanner.find_nodes([self._bound_args, self._bound_kwargs, self._bound_other_args_to_resolve]):
        if node not in replace_table:
            replace_table[node] = fn(node)
    new_args, new_kwargs, new_other_args_to_resolve = scanner.replace_nodes(replace_table)
    scanner.clear()
    return self._copy(new_args, new_kwargs, self.get_options(), new_other_args_to_resolve)