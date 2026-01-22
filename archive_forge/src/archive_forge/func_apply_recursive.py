import ray
from ray.dag.base import DAGNodeBase
from ray.dag.py_obj_scanner import _PyObjScanner
from ray.util.annotations import DeveloperAPI
from typing import (
import uuid
import asyncio
def apply_recursive(self, fn: 'Callable[[DAGNode], T]') -> T:
    """Apply callable on each node in this DAG in a bottom-up tree walk.

        Args:
            fn: Callable that will be applied once to each node in the
                DAG. It will be applied recursively bottom-up, so nodes can
                assume the fn has been applied to their args already.

        Returns:
            Return type of the fn after application to the tree.
        """
    if not type(fn).__name__ == '_CachingFn':

        class _CachingFn:

            def __init__(self, fn):
                self.cache = {}
                self.fn = fn
                self.fn.cache = self.cache
                self.input_node_uuid = None

            def __call__(self, node: 'DAGNode'):
                if node._stable_uuid not in self.cache:
                    self.cache[node._stable_uuid] = self.fn(node)
                if type(node).__name__ == 'InputNode':
                    if not self.input_node_uuid:
                        self.input_node_uuid = node._stable_uuid
                    elif self.input_node_uuid != node._stable_uuid:
                        raise AssertionError('Each DAG should only have one unique InputNode.')
                return self.cache[node._stable_uuid]
        fn = _CachingFn(fn)
    elif self._stable_uuid in fn.cache:
        return fn.cache[self._stable_uuid]
    return fn(self._apply_and_replace_all_child_nodes(lambda node: node.apply_recursive(fn)))