import gc
import platform
from typing import Iterable
from .metrics_core import CounterMetricFamily, Metric
from .registry import Collector, CollectorRegistry, REGISTRY
Collector for Garbage collection statistics.