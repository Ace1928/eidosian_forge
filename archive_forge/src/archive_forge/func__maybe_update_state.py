import time
from collections import defaultdict
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import ray
from ray.air.execution.resources.request import (
from ray.air.execution.resources.resource_manager import ResourceManager
from ray.util.annotations import DeveloperAPI
from ray.util.placement_group import PlacementGroup, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
def _maybe_update_state(self):
    now = time.monotonic()
    if now > self._last_update + self.update_interval_s:
        self.update_state()