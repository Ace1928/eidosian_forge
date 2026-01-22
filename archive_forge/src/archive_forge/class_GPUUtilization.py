import logging
import threading
from collections import deque
from typing import TYPE_CHECKING, List
from wandb.vendor.pynvml import pynvml
from .aggregators import aggregate_mean
from .asset_registry import asset_registry
from .interfaces import Interface, Metric, MetricsMonitor
class GPUUtilization:
    """GPU utilization in percent for each GPU."""
    name = 'gpu.{}.gpu'
    samples: 'Deque[List[float]]'

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.samples = deque([])

    def sample(self) -> None:
        gpu_utilization_rate = []
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            gpu_utilization_rate.append(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        self.samples.append(gpu_utilization_rate)

    def clear(self) -> None:
        self.samples.clear()

    def aggregate(self) -> dict:
        if not self.samples:
            return {}
        stats = {}
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            samples = [sample[i] for sample in self.samples]
            aggregate = aggregate_mean(samples)
            stats[self.name.format(i)] = aggregate
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            if gpu_in_use_by_this_process(handle, self.pid):
                stats[self.name.format(f'process.{i}')] = aggregate
        return stats