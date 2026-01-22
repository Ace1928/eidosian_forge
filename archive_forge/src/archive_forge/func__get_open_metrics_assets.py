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
def _get_open_metrics_assets(self) -> List['Asset']:
    open_metrics_endpoints = self.settings._stats_open_metrics_endpoints
    if not open_metrics_endpoints:
        return []
    assets: List[Asset] = []
    for name, endpoint in open_metrics_endpoints.items():
        if not OpenMetrics.is_available(url=endpoint):
            continue
        logger.debug(f'Monitoring OpenMetrics endpoint: {endpoint}')
        open_metrics = OpenMetrics(interface=self.asset_interface or self.backend_interface, settings=self.settings, shutdown_event=self._shutdown_event, name=name, url=endpoint)
        assets.append(open_metrics)
    return assets