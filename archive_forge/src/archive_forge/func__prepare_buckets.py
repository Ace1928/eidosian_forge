import os
from threading import Lock
import time
import types
from typing import (
import warnings
from . import values  # retain this import style for testability
from .context_managers import ExceptionCounter, InprogressTracker, Timer
from .metrics_core import (
from .registry import Collector, CollectorRegistry, REGISTRY
from .samples import Exemplar, Sample
from .utils import floatToGoString, INF
def _prepare_buckets(self, source_buckets: Sequence[Union[float, str]]) -> None:
    buckets = [float(b) for b in source_buckets]
    if buckets != sorted(buckets):
        raise ValueError('Buckets not in sorted order')
    if buckets and buckets[-1] != INF:
        buckets.append(INF)
    if len(buckets) < 2:
        raise ValueError('Must have at least two buckets')
    self._upper_bounds = buckets