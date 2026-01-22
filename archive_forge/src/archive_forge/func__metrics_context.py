import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
@contextmanager
def _metrics_context(self):
    self.thread_local.metrics = self.shared_metrics.get()
    yield