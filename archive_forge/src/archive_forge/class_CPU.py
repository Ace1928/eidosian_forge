import threading
from collections import deque
from typing import TYPE_CHECKING, List, Optional
from .aggregators import aggregate_last, aggregate_mean
from .asset_registry import asset_registry
from .interfaces import Interface, Metric, MetricsMonitor
@asset_registry.register
class CPU:

    def __init__(self, interface: 'Interface', settings: 'SettingsStatic', shutdown_event: threading.Event) -> None:
        self.name: str = self.__class__.__name__.lower()
        self.metrics: List[Metric] = [ProcessCpuPercent(settings._stats_pid), CpuPercent(), ProcessCpuThreads(settings._stats_pid)]
        self.metrics_monitor: MetricsMonitor = MetricsMonitor(self.name, self.metrics, interface, settings, shutdown_event)

    @classmethod
    def is_available(cls) -> bool:
        return psutil is not None

    def probe(self) -> dict:
        asset_info = {'cpu_count': psutil.cpu_count(logical=False), 'cpu_count_logical': psutil.cpu_count(logical=True)}
        try:
            asset_info['cpu_freq'] = {'current': psutil.cpu_freq().current, 'min': psutil.cpu_freq().min, 'max': psutil.cpu_freq().max}
            asset_info['cpu_freq_per_core'] = [{'current': freq.current, 'min': freq.min, 'max': freq.max} for freq in psutil.cpu_freq(percpu=True)]
        except Exception:
            pass
        return asset_info

    def start(self) -> None:
        self.metrics_monitor.start()

    def finish(self) -> None:
        self.metrics_monitor.finish()