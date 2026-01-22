import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List
from ray.autoscaler.v2.instance_manager.instance_storage import (
from ray.autoscaler.v2.instance_manager.node_provider import NodeProvider
from ray.core.generated.instance_manager_pb2 import Instance
def _periodic_reconcile_helper(self) -> None:
    try:
        self._reconcile_with_node_provider()
    except Exception:
        logger.exception('Failed to reconcile with node provider')
    with self._reconcile_timer_lock:
        self._reconcile_timer = threading.Timer(self._reconcile_interval_s, self._periodic_reconcile_helper)