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
class _DelayedDeserialization:

    def __init__(self, serialized: bytes):
        self._serialized = serialized

    def __reduce__(self):
        return (cloudpickle.loads, (self._serialized,))