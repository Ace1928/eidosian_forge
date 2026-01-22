import ray
from ray.dag.base import DAGNodeBase
from ray.dag.py_obj_scanner import _PyObjScanner
from ray.util.annotations import DeveloperAPI
from typing import (
import uuid
import asyncio
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