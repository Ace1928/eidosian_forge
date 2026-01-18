import abc
import functools
import inspect
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
import ray
from ray.actor import ActorHandle
from ray.air._internal.util import StartTraceback, find_free_port
from ray.exceptions import RayActorError
from ray.types import ObjectRef
def update_env_vars(env_vars: Dict[str, Any]):
    """Updates the environment variables on this worker process.

    Args:
        env_vars: Environment variables to set.
    """
    sanitized = {k: str(v) for k, v in env_vars.items()}
    os.environ.update(sanitized)