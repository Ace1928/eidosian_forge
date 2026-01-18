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
def update_workflow_status(self, status: WorkflowStatus):
    """Update the status of the workflow.
        This method is NOT thread-safe. It is handled by the workflow management actor.
        """
    self._status_storage.update_workflow_status(self._workflow_id, status)
    if status == WorkflowStatus.RUNNING:
        self._put(self._key_workflow_prerun_metadata(), {'start_time': time.time()}, True)
    elif status in (WorkflowStatus.SUCCESSFUL, WorkflowStatus.FAILED):
        self._put(self._key_workflow_postrun_metadata(), {'end_time': time.time()}, True)