import functools
import logging
import os
import platform
import queue
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Set, Type
import ray
from ray.air._internal.session import _get_session
from ray.air._internal.util import RunnerThread, StartTraceback
from ray.air.constants import (
from ray.data import Dataset
from ray.train import Checkpoint
from ray.train._internal.accelerator import Accelerator
from ray.train._internal.storage import StorageContext
from ray.train.constants import (
from ray.train.error import SessionMisuseError
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.debug import log_once
from ray.util.placement_group import _valid_resource_shape
from ray.util.scheduling_strategies import (
def _warn_session_misuse(default_value: Any=None):
    """Warns if fn is being used outside of session and returns ``default_value``."""

    def inner(fn: Callable):
        fn_name = fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            session = _get_session()
            if not session:
                if log_once(f'{SESSION_MISUSE_LOG_ONCE_KEY}-{fn_name}'):
                    warnings.warn(f'`{fn_name}` is meant to only be called inside a function that is executed by a Tuner or Trainer. Returning `{default_value}`.')
                return default_value
            return fn(*args, **kwargs)
        return wrapper
    return inner