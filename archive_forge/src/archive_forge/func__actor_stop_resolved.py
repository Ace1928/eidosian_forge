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
def _actor_stop_resolved(self, tracked_actor: TrackedActor):
    """Callback to be invoked when actor stopped"""
    self._cleanup_actor(tracked_actor=tracked_actor)
    if tracked_actor._on_stop:
        tracked_actor._on_stop(tracked_actor)