import base64
import logging
from collections import defaultdict
from enum import Enum
from typing import List
import ray
from ray._private.internal_api import node_stats
from ray._raylet import ActorID, JobID, TaskID
class GroupByType(Enum):
    NODE_ADDRESS = 'node'
    STACK_TRACE = 'stack_trace'