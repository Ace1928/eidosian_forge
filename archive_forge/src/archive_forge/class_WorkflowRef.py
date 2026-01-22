import base64
import json
from ray import cloudpickle
from enum import Enum, unique
import hashlib
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
import ray
from ray import ObjectRef
from ray._private.utils import get_or_create_event_loop
from ray.util.annotations import PublicAPI
@dataclass
class WorkflowRef:
    """This class represents a reference of a workflow output.

    A reference means the workflow has already been executed,
    and we have both the workflow task ID and the object ref to it
    living outputs.

    This could be used when you want to return a running workflow
    from a workflow task. For example, the remaining workflows
    returned by 'workflow.wait' contains a static ref to these
    pending workflows.
    """
    task_id: TaskID
    ref: Optional[ObjectRef] = None

    @classmethod
    def from_output(cls, task_id: str, output: Any):
        """Create static ref from given output."""
        if not isinstance(output, cls):
            if not isinstance(output, ray.ObjectRef):
                output = ray.put(output)
            output = cls(task_id=task_id, ref=output)
        return output

    def __hash__(self):
        return hash(self.task_id)