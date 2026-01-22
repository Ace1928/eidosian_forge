import math
import threading
import time
from typing import Dict, List
import ray
from ray.data.context import DataContext
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def _test_set_timeout(self, ttl):
    """Set the timeout. This is for test only"""
    self._timeout = ttl