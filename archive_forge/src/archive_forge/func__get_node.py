import copy
import logging
import time
from functools import wraps
from threading import RLock
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple
import googleapiclient
from ray.autoscaler._private.gcp.config import (
from ray.autoscaler._private.gcp.node import GCPTPU  # noqa
from ray.autoscaler._private.gcp.node import (
from ray.autoscaler._private.gcp.tpu_command_runner import TPUCommandRunner
from ray.autoscaler.command_runner import CommandRunnerInterface
from ray.autoscaler.node_provider import NodeProvider
@_retry
def _get_node(self, node_id: str) -> GCPNode:
    self.non_terminated_nodes({})
    with self.lock:
        if node_id in self.cached_nodes:
            return self.cached_nodes[node_id]
        resource = self._get_resource_depending_on_node_name(node_id)
        instance = resource.get_instance(node_id=node_id)
        return instance