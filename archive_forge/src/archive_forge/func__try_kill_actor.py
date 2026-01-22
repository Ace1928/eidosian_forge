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
def _try_kill_actor(self) -> bool:
    """Try to kill actor scheduled for termination."""
    if not self._live_actors_to_kill:
        return False
    tracked_actor = self._live_actors_to_kill.pop()
    ray_actor, acquired_resources = self._live_actors_to_ray_actors_resources[tracked_actor]
    ray.kill(ray_actor)
    self._cleanup_actor_futures(tracked_actor)
    self._actor_stop_resolved(tracked_actor)
    return True