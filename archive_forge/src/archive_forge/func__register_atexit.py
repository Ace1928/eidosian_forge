import atexit
import logging
from functools import partial
from types import FunctionType
from typing import Callable, Optional, Type, Union
import ray
import ray.cloudpickle as pickle
from ray.experimental.internal_kv import (
from ray.tune.error import TuneError
from ray.util.annotations import DeveloperAPI
def _register_atexit(self):
    if self._atexit_handler_registered:
        return
    if ray._private.worker.global_worker.mode != ray.SCRIPT_MODE:
        return
    atexit.register(_unregister_all)
    self._atexit_handler_registered = True