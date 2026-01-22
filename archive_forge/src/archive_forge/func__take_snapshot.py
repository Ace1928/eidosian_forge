from collections import defaultdict, namedtuple
import gc
import os
import re
import time
import tracemalloc
from typing import Callable, List, Optional
from ray.util.annotations import DeveloperAPI
def _take_snapshot(table, suspicious=None):
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('traceback')
    for stat in top_stats[:100]:
        if not suspicious or stat.traceback in suspicious:
            table[stat.traceback].append(stat.size)