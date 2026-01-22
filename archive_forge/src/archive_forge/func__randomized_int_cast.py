import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
def _randomized_int_cast(float_value):
    base = int(float_value)
    remainder = float_value - base
    if random.random() < remainder:
        base += 1
    return base