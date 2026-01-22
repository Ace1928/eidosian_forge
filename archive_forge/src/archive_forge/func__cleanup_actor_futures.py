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
def _cleanup_actor_futures(self, tracked_actor: TrackedActor):
    self.clear_actor_task_futures(tracked_actor=tracked_actor)
    futures = self._tracked_actors_to_state_futures.pop(tracked_actor, [])
    for future in futures:
        self._actor_state_events.discard_future(future)