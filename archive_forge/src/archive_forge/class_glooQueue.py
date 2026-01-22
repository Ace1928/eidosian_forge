import asyncio
import time
from typing import List
import numpy
import ray
import ray.experimental.internal_kv as internal_kv
from ray._raylet import GcsClient
from ray.util.collective.types import ReduceOp, torch_available
from ray.util.queue import _QueueActor
class glooQueue(_QueueActor):

    def index(self, group_name):
        try:
            return self.queue._queue.index(group_name)
        except ValueError:
            return -1