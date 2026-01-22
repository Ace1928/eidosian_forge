import base64
import logging
from collections import defaultdict
from enum import Enum
from typing import List
import ray
from ray._private.internal_api import node_stats
from ray._raylet import ActorID, JobID, TaskID
def _sort_by(self, sorting_type: SortingType):
    if sorting_type == SortingType.PID:
        self.table.sort(key=lambda entry: entry.pid)
    elif sorting_type == SortingType.OBJECT_SIZE:
        self.table.sort(key=lambda entry: entry.object_size)
    elif sorting_type == SortingType.REFERENCE_TYPE:
        self.table.sort(key=lambda entry: entry.reference_type)
    else:
        raise ValueError(f'Give sorting type: {sorting_type} is invalid.')
    return self