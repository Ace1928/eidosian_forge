import base64
import logging
from collections import defaultdict
from enum import Enum
from typing import List
import ray
from ray._private.internal_api import node_stats
from ray._raylet import ActorID, JobID, TaskID
def _get_reference_type(self) -> str:
    if self._is_object_ref_actor_handle():
        return ReferenceType.ACTOR_HANDLE.value
    if self.pinned_in_memory:
        return ReferenceType.PINNED_IN_MEMORY.value
    elif self.submitted_task_ref_count > 0:
        return ReferenceType.USED_BY_PENDING_TASK.value
    elif self.local_ref_count > 0:
        return ReferenceType.LOCAL_REFERENCE.value
    elif len(self.contained_in_owned) > 0:
        return ReferenceType.CAPTURED_IN_OBJECT.value
    else:
        return ReferenceType.UNKNOWN_STATUS.value