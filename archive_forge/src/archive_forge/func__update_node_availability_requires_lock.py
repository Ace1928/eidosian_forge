import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
from ray.autoscaler._private.constants import (
from ray.autoscaler.node_launch_exception import NodeLaunchException
def _update_node_availability_requires_lock(self, node_type: str, timestamp: int, node_launch_exception: Optional[NodeLaunchException]) -> None:
    if node_launch_exception is None:
        record = NodeAvailabilityRecord(node_type=node_type, is_available=True, last_checked_timestamp=timestamp, unavailable_node_information=None)
    else:
        info = UnavailableNodeInformation(category=node_launch_exception.category, description=node_launch_exception.description)
        record = NodeAvailabilityRecord(node_type=node_type, is_available=False, last_checked_timestamp=timestamp, unavailable_node_information=info)
    expiration_time = timestamp + self.ttl
    self.store[node_type] = (expiration_time, record)
    self._remove_old_entries()