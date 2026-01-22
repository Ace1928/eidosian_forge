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
def _validate_labelname(l):
    if not METRIC_LABEL_NAME_RE.match(l):
        raise ValueError('Invalid label metric name: ' + l)
    if RESERVED_METRIC_LABEL_NAME_RE.match(l):
        raise ValueError('Reserved label metric name: ' + l)