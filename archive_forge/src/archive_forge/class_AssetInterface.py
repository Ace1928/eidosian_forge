import datetime
import logging
import queue
import threading
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Deque, Dict, List, Optional, Tuple
from .assets.asset_registry import asset_registry
from .assets.interfaces import Asset, Interface
from .assets.open_metrics import OpenMetrics
from .system_info import SystemInfo
class AssetInterface:

    def __init__(self) -> None:
        self.metrics_queue: queue.Queue[dict] = queue.Queue()
        self.telemetry_queue: queue.Queue[TelemetryRecord] = queue.Queue()

    def publish_stats(self, stats: dict) -> None:
        self.metrics_queue.put(stats)

    def _publish_telemetry(self, telemetry: 'TelemetryRecord') -> None:
        self.telemetry_queue.put(telemetry)

    def publish_files(self, files_dict: 'FilesDict') -> None:
        pass