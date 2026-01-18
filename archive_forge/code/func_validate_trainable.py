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
@DeveloperAPI
def validate_trainable(trainable_name):
    if not _has_trainable(trainable_name):
        from ray.rllib import _register_all
        _register_all()
        if not _has_trainable(trainable_name):
            raise TuneError('Unknown trainable: ' + trainable_name)