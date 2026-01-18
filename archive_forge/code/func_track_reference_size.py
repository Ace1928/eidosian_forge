import base64
import logging
from collections import defaultdict
from enum import Enum
from typing import List
import ray
from ray._private.internal_api import node_stats
from ray._raylet import ActorID, JobID, TaskID
def track_reference_size(group):
    """Returns dictionary mapping reference type
    to memory usage for a given memory table group."""
    d = defaultdict(int)
    table_name = {'LOCAL_REFERENCE': 'total_local_ref_count', 'PINNED_IN_MEMORY': 'total_pinned_in_memory', 'USED_BY_PENDING_TASK': 'total_used_by_pending_task', 'CAPTURED_IN_OBJECT': 'total_captured_in_objects', 'ACTOR_HANDLE': 'total_actor_handles'}
    for entry in group['entries']:
        size = entry['object_size']
        if size == -1:
            size = 0
        d[table_name[entry['reference_type']]] += size
    return d