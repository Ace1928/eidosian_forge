import logging
import random
import time
import uuid
from collections import defaultdict, Counter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import ray
from ray.air.execution._internal.event_manager import RayEventManager
from ray.air.execution.resources import (
from ray.air.execution._internal.tracked_actor import TrackedActor
from ray.air.execution._internal.tracked_actor_task import TrackedActorTask
from ray.exceptions import RayTaskError, RayActorError
def _handle_ready_resource_future(self):
    """Handle a resource future that became ready.

        - Update state of the resource manager
        - Try to start one actor
        """
    self._resource_manager.update_state()
    self._try_start_actors(max_actors=1)