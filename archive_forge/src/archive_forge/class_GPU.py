import logging
import threading
from collections import deque
from typing import TYPE_CHECKING, List
from wandb.vendor.pynvml import pynvml
from .aggregators import aggregate_mean
from .asset_registry import asset_registry
from .interfaces import Interface, Metric, MetricsMonitor
@asset_registry.register
class GPU:

    def __init__(self, interface: 'Interface', settings: 'SettingsStatic', shutdown_event: threading.Event) -> None:
        self.name = self.__class__.__name__.lower()
        self.metrics: List[Metric] = [GPUMemoryAllocated(settings._stats_pid), GPUMemoryAllocatedBytes(settings._stats_pid), GPUMemoryUtilization(settings._stats_pid), GPUUtilization(settings._stats_pid), GPUTemperature(settings._stats_pid), GPUPowerUsageWatts(settings._stats_pid), GPUPowerUsagePercent(settings._stats_pid)]
        self.metrics_monitor = MetricsMonitor(self.name, self.metrics, interface, settings, shutdown_event)

    @classmethod
    def is_available(cls) -> bool:
        try:
            pynvml.nvmlInit()
            return True
        except pynvml.NVMLError_LibraryNotFound:
            return False
        except Exception as e:
            logger.error(f'Error initializing NVML: {e}')
            return False

    def start(self) -> None:
        self.metrics_monitor.start()

    def finish(self) -> None:
        self.metrics_monitor.finish()

    def probe(self) -> dict:
        info = {}
        try:
            pynvml.nvmlInit()
            info['gpu'] = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
            info['gpu_count'] = pynvml.nvmlDeviceGetCount()
            device_count = pynvml.nvmlDeviceGetCount()
            devices = []
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                gpu_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                devices.append({'name': pynvml.nvmlDeviceGetName(handle), 'memory_total': gpu_info.total})
            info['gpu_devices'] = devices
        except pynvml.NVMLError:
            pass
        return info