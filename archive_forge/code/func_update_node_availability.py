import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
from ray.autoscaler._private.constants import (
from ray.autoscaler.node_launch_exception import NodeLaunchException
def update_node_availability(self, node_type: str, timestamp: int, node_launch_exception: Optional[NodeLaunchException]) -> None:
    """
        Update the availability and details of a single ndoe type.

        Args:
          node_type: The node type.
          timestamp: The timestamp that this information is accurate as of.
          node_launch_exception: Details about why the node launch failed. If
            empty, the node type will be considered available."""
    with self.lock:
        self._update_node_availability_requires_lock(node_type, timestamp, node_launch_exception)