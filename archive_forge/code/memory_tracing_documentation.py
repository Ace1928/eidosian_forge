from io import StringIO
from typing import Dict, List
import ray
from ray.data.context import DataContext
Record that an object has been deleted (and delete if free=True).

    Args:
        ref: The object we no longer need.
        loc: A human-readable string identifying the call site.
        free: Whether to eagerly destroy the object instead of waiting for Ray
            reference counting to kick in.
    