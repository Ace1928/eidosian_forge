import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar
import ray
import ray._private.ray_constants as ray_constants
from ray._private.ray_constants import env_integer
from ray.data import Dataset
from ray.exceptions import RayActorError
from ray.train import Checkpoint, DataConfig
from ray.train._internal.session import (
from ray.train._internal.storage import StorageContext
from ray.train._internal.utils import check_for_failure
from ray.train._internal.worker_group import WorkerGroup
from ray.train.backend import BackendConfig
from ray.train.constants import (
from ray.util.placement_group import get_current_placement_group, remove_placement_group
def _increment_failures(self):
    self._num_failures += 1
    if self._num_failures >= self._max_failures:
        failure = self._last_failure
        self._last_failure = None
        if self._max_failures > 0:
            exc = RuntimeError(f'Training has failed after {self._num_failures} attempts.')
            raise exc.with_traceback(None) from failure
        else:
            raise failure