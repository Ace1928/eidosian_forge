import dis
import hashlib
import importlib
import inspect
import json
import logging
import os
import threading
import time
import traceback
from collections import defaultdict, namedtuple
from typing import Optional, Callable
import ray
import ray._private.profiling as profiling
from ray import cloudpickle as pickle
from ray._private import ray_constants
from ray._private.inspect_util import (
from ray._private.ray_constants import KV_NAMESPACE_FUNCTION_TABLE
from ray._private.utils import (
from ray._private.serialization import pickle_dumps
from ray._raylet import (
def _load_function_from_local(self, function_descriptor):
    assert not function_descriptor.is_actor_method()
    function_id = function_descriptor.function_id
    module_name, function_name = (function_descriptor.module_name, function_descriptor.function_name)
    object = self.load_function_or_class_from_local(module_name, function_name)
    if object is not None:
        function = object._function
        self._function_execution_info[function_id] = FunctionExecutionInfo(function=function, function_name=function_name, max_calls=0)
        self._num_task_executions[function_id] = 0
        return True
    else:
        return False