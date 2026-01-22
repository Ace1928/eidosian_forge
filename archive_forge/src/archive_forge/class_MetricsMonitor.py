import datetime
import logging
import sys
import threading
from typing import TYPE_CHECKING, Any, List, Optional, TypeVar
import psutil
class MetricsMonitor:
    """Takes care of collecting, sampling, serializing, and publishing a set of metrics."""

    def __init__(self, asset_name: str, metrics: List[Metric], interface: Interface, settings: 'SettingsStatic', shutdown_event: threading.Event) -> None:
        self.metrics = metrics
        self.asset_name = asset_name
        self._interface = interface
        self._process: Optional[threading.Thread] = None
        self._shutdown_event: threading.Event = shutdown_event
        self.sampling_interval: float = float(max(0.1, settings._stats_sample_rate_seconds))
        self.samples_to_aggregate: int = min(30, max(1, settings._stats_samples_to_average))

    def monitor(self) -> None:
        """Poll the Asset metrics."""
        while not self._shutdown_event.is_set():
            for _ in range(self.samples_to_aggregate):
                for metric in self.metrics:
                    try:
                        metric.sample()
                    except psutil.NoSuchProcess:
                        logger.info(f'Process {metric.name} has exited.')
                        self._shutdown_event.set()
                        break
                    except Exception as e:
                        logger.error(f'Failed to sample metric: {e}')
                self._shutdown_event.wait(self.sampling_interval)
                if self._shutdown_event.is_set():
                    break
            self.publish()

    def aggregate(self) -> dict:
        """Return a dict of metrics."""
        aggregated_metrics = {}
        for metric in self.metrics:
            try:
                serialized_metric = metric.aggregate()
                aggregated_metrics.update(serialized_metric)
            except Exception as e:
                logger.error(f'Failed to serialize metric: {e}')
        return aggregated_metrics

    def publish(self) -> None:
        """Publish the Asset metrics."""
        try:
            aggregated_metrics = self.aggregate()
            if aggregated_metrics:
                self._interface.publish_stats(aggregated_metrics)
            for metric in self.metrics:
                metric.clear()
        except Exception as e:
            logger.error(f'Failed to publish metrics: {e}')

    def start(self) -> None:
        if self._process is not None or self._shutdown_event.is_set():
            return None
        thread_name = f'{self.asset_name[:15]}'
        try:
            for metric in self.metrics:
                if isinstance(metric, SetupTeardown):
                    metric.setup()
            self._process = threading.Thread(target=self.monitor, daemon=True, name=thread_name)
            self._process.start()
            logger.info(f'Started {thread_name} monitoring')
        except Exception as e:
            logger.warning(f'Failed to start {thread_name} monitoring: {e}')
            self._process = None

    def finish(self) -> None:
        if self._process is None:
            return None
        thread_name = f'{self.asset_name[:15]}'
        try:
            self._process.join()
            logger.info(f'Joined {thread_name} monitor')
            for metric in self.metrics:
                if isinstance(metric, SetupTeardown):
                    metric.teardown()
        except Exception as e:
            logger.warning(f'Failed to finish {thread_name} monitoring: {e}')
        finally:
            self._process = None