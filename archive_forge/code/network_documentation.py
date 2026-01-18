import threading
from collections import deque
from typing import TYPE_CHECKING, List
from .aggregators import aggregate_mean
from .asset_registry import asset_registry
from .interfaces import Interface, Metric, MetricsMonitor
Return a dict of the hardware information.