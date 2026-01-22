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
def _build_full_name(metric_type, name, namespace, subsystem, unit):
    full_name = ''
    if namespace:
        full_name += namespace + '_'
    if subsystem:
        full_name += subsystem + '_'
    full_name += name
    if metric_type == 'counter' and full_name.endswith('_total'):
        full_name = full_name[:-6]
    if unit and (not full_name.endswith('_' + unit)):
        full_name += '_' + unit
    if unit and metric_type in ('info', 'stateset'):
        raise ValueError('Metric name is of a type that cannot have a unit: ' + full_name)
    return full_name