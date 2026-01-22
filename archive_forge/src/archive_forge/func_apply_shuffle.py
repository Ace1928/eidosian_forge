import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
def apply_shuffle(it):
    buffer = []
    for item in it:
        if isinstance(item, _NextValueNotReady):
            yield item
        else:
            buffer.append(item)
            if len(buffer) >= shuffle_buffer_size:
                yield buffer.pop(shuffle_random.randint(0, len(buffer) - 1))
    while len(buffer) > 0:
        yield buffer.pop(shuffle_random.randint(0, len(buffer) - 1))