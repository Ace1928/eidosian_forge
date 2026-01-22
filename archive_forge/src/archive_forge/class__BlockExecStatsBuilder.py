import collections
import os
import time
from dataclasses import dataclass
from typing import (
import numpy as np
import ray
from ray import DynamicObjectRefGenerator
from ray.data._internal.util import _check_pyarrow_version, _truncated_repr
from ray.types import ObjectRef
from ray.util.annotations import DeveloperAPI
import psutil
class _BlockExecStatsBuilder:
    """Helper class for building block stats.

    When this class is created, we record the start time. When build() is
    called, the time delta is saved as part of the stats.
    """

    def __init__(self):
        self.start_time = time.perf_counter()
        self.start_cpu = time.process_time()

    def build(self) -> 'BlockExecStats':
        self.end_time = time.perf_counter()
        self.end_cpu = time.process_time()
        stats = BlockExecStats()
        stats.start_time_s = self.start_time
        stats.end_time_s = self.end_time
        stats.wall_time_s = self.end_time - self.start_time
        stats.cpu_time_s = self.end_cpu - self.start_cpu
        if resource is None:
            process = psutil.Process(os.getpid())
            stats.max_rss_bytes = int(process.memory_info().rss)
        else:
            stats.max_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1000.0)
        return stats