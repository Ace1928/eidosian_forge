import base64
import logging
from collections import defaultdict
from enum import Enum
from typing import List
import ray
from ray._private.internal_api import node_stats
from ray._raylet import ActorID, JobID, TaskID
class ReferenceType(Enum):
    ACTOR_HANDLE = 'ACTOR_HANDLE'
    PINNED_IN_MEMORY = 'PINNED_IN_MEMORY'
    LOCAL_REFERENCE = 'LOCAL_REFERENCE'
    USED_BY_PENDING_TASK = 'USED_BY_PENDING_TASK'
    CAPTURED_IN_OBJECT = 'CAPTURED_IN_OBJECT'
    UNKNOWN_STATUS = 'UNKNOWN_STATUS'