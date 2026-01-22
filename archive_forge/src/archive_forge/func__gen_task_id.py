import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import ray
from ray import cloudpickle
from ray._private import storage
from ray.types import ObjectRef
from ray.workflow.common import (
from ray.workflow.exceptions import WorkflowNotFoundError
from ray.workflow import workflow_context
from ray.workflow import serialization
from ray.workflow import serialization_context
from ray.workflow.workflow_state import WorkflowExecutionState
from ray.workflow.storage import DataLoadError, DataSaveError, KeyNotFoundError
def _gen_task_id():
    key = self._key_num_tasks_with_name(task_name)
    try:
        val = self._get(key, True)
        self._put(key, val + 1, True)
        return val + 1
    except KeyNotFoundError:
        self._put(key, 0, True)
        return 0