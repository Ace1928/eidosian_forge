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
class _ParameterRegistry:

    def __init__(self):
        self.to_flush = {}
        self.references = {}

    def put(self, k, v):
        self.to_flush[k] = v
        if ray.is_initialized():
            self.flush()

    def get(self, k):
        if not ray.is_initialized():
            return self.to_flush[k]
        return ray.get(self.references[k])

    def flush(self):
        for k, v in self.to_flush.items():
            if isinstance(v, ray.ObjectRef):
                self.references[k] = v
            else:
                self.references[k] = ray.put(v)
        self.to_flush.clear()