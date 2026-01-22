import inspect
import logging
import types
from typing import Any, Callable, Dict, Optional, Type, Union, TYPE_CHECKING
import ray
from ray.tune.execution.placement_groups import (
from ray.air.config import ScalingConfig
from ray.tune.registry import _ParameterRegistry
from ray.tune.utils import _detect_checkpoint_function
from ray.util.annotations import PublicAPI
def _trainable(config):
    return trainable(config)