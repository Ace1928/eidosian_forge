import threading
from collections import deque
from typing import TYPE_CHECKING, List, Optional
from .aggregators import aggregate_last, aggregate_mean
from .asset_registry import asset_registry
from .interfaces import Interface, Metric, MetricsMonitor
class ProcessCpuPercent:
    """CPU usage of the process in percent normalized by the number of CPUs."""
    name = 'cpu'

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.samples: Deque[float] = deque([])
        self.process: Optional[psutil.Process] = None

    def sample(self) -> None:
        if self.process is None:
            self.process = psutil.Process(self.pid)
        self.samples.append(self.process.cpu_percent() / psutil.cpu_count())

    def clear(self) -> None:
        self.samples.clear()

    def aggregate(self) -> dict:
        if not self.samples:
            return {}
        aggregate = aggregate_mean(self.samples)
        return {self.name: aggregate}