import threading
from collections import deque
from typing import TYPE_CHECKING, List, Optional
from .aggregators import aggregate_mean
from .asset_registry import asset_registry
from .interfaces import Interface, Metric, MetricsMonitor
class MemoryAvailable:
    """Total system memory available in MB."""
    name = 'proc.memory.availableMB'
    samples: 'Deque[float]'

    def __init__(self) -> None:
        self.samples = deque([])

    def sample(self) -> None:
        self.samples.append(psutil.virtual_memory().available / 1024 / 1024)

    def clear(self) -> None:
        self.samples.clear()

    def aggregate(self) -> dict:
        if not self.samples:
            return {}
        aggregate = aggregate_mean(self.samples)
        return {self.name: aggregate}