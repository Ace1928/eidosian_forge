import threading
from collections import deque
from typing import TYPE_CHECKING, List, Optional
from wandb.errors.term import termwarn
from .aggregators import aggregate_mean
from .asset_registry import asset_registry
from .interfaces import Interface, Metric, MetricsMonitor
class DiskOut:
    """Total system disk write in MB."""
    name = 'disk.out'
    samples: 'Deque[float]'

    def __init__(self) -> None:
        self.samples = deque([])
        self.write_init: Optional[int] = None

    def sample(self) -> None:
        if self.write_init is None:
            self.write_init = psutil.disk_io_counters().write_bytes
        self.samples.append((psutil.disk_io_counters().write_bytes - self.write_init) / 1024 / 1024)

    def clear(self) -> None:
        self.samples.clear()

    def aggregate(self) -> dict:
        if not self.samples:
            return {}
        aggregate = aggregate_mean(self.samples)
        return {self.name: aggregate}