import ray
from ray.dag.dag_node import DAGNode
from ray.dag.input_node import InputNode
from ray.dag.format_utils import get_dag_node_str
from ray.dag.constants import (
from ray.util.annotations import DeveloperAPI
from typing import Any, Dict, List, Optional, Tuple
class _UnboundClassMethodNode(object):

    def __init__(self, actor: ClassNode, method_name: str):
        self._actor = actor
        self._method_name = method_name
        self._options = {}

    def bind(self, *args, **kwargs):
        other_args_to_resolve = {PARENT_CLASS_NODE_KEY: self._actor, PREV_CLASS_METHOD_CALL_KEY: self._actor._last_call}
        node = ClassMethodNode(self._method_name, args, kwargs, self._options, other_args_to_resolve=other_args_to_resolve)
        self._actor._last_call = node
        return node

    def __getattr__(self, attr: str):
        if attr == 'remote':
            raise AttributeError('.remote() cannot be used on ClassMethodNodes. Use .bind() instead to express an symbolic actor call.')
        else:
            return self.__getattribute__(attr)

    def options(self, **options):
        self._options = options
        return self