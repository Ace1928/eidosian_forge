import json
import logging
import pathlib
import platform
import subprocess
import sys
import threading
from collections import deque
from typing import TYPE_CHECKING, List
from wandb.sdk.lib import telemetry
from .aggregators import aggregate_mean
from .asset_registry import asset_registry
from .interfaces import Interface, Metric, MetricsMonitor
@asset_registry.register
class GPUApple:

    def __init__(self, interface: 'Interface', settings: 'SettingsStatic', shutdown_event: threading.Event) -> None:
        self.name = self.__class__.__name__.lower()
        self.metrics: List[Metric] = [GPUAppleStats()]
        self.metrics_monitor = MetricsMonitor(self.name, self.metrics, interface, settings, shutdown_event)
        telemetry_record = telemetry.TelemetryRecord()
        telemetry_record.env.m1_gpu = True
        interface._publish_telemetry(telemetry_record)

    @classmethod
    def is_available(cls) -> bool:
        return platform.system() == 'Darwin' and platform.processor() == 'arm'

    def start(self) -> None:
        self.metrics_monitor.start()

    def finish(self) -> None:
        self.metrics_monitor.finish()

    def probe(self) -> dict:
        return {self.name: {'type': 'arm', 'vendor': 'Apple'}}