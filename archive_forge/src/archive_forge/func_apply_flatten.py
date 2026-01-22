import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
def apply_flatten(it):
    for item in it:
        if isinstance(item, _NextValueNotReady):
            yield item
        else:
            for subitem in item:
                yield subitem