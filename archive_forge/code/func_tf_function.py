import logging
import numpy as np
import os
import sys
from typing import Any, Optional
from ray.rllib.utils.annotations import DeveloperAPI, PublicAPI
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.typing import TensorShape, TensorType
@DeveloperAPI
def tf_function(tf_module):
    """Conditional decorator for @tf.function.

    Use @tf_function(tf) instead to avoid errors if tf is not installed."""

    def decorator(func):
        if tf_module is None or tf_module.executing_eagerly():
            return func
        return tf_module.function(func)
    return decorator