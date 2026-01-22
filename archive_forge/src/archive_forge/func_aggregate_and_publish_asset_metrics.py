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
def aggregate_and_publish_asset_metrics(self) -> None:
    if self.asset_interface is None:
        return None
    size = self.asset_interface.metrics_queue.qsize()
    aggregated_metrics = {}
    for _ in range(size):
        item = self.asset_interface.metrics_queue.get()
        aggregated_metrics.update(item)
    if aggregated_metrics:
        t = datetime.datetime.now().timestamp()
        for k, v in aggregated_metrics.items():
            self.buffer[k].append((t, v))
        self.backend_interface.publish_stats(aggregated_metrics)