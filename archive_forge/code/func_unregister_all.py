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
def unregister_all(self, category: Optional[str]=None):
    remaining = set()
    for cat, key in self._registered:
        if category and category == cat:
            self.unregister(cat, key)
        else:
            remaining.add((cat, key))
    self._registered = remaining