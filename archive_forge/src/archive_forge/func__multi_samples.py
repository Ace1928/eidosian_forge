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
def _multi_samples(self) -> Iterable[Sample]:
    with self._lock:
        metrics = self._metrics.copy()
    for labels, metric in metrics.items():
        series_labels = list(zip(self._labelnames, labels))
        for suffix, sample_labels, value, timestamp, exemplar in metric._samples():
            yield Sample(suffix, dict(series_labels + list(sample_labels.items())), value, timestamp, exemplar)