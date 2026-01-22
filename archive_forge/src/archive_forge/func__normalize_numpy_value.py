import abc
import hashlib
import json
import random
import time
import numpy as np
from tensorboard.compat import tf2 as tf
from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import metadata
from tensorboard.plugins.hparams import plugin_data_pb2
def _normalize_numpy_value(value):
    """Convert a Python or Numpy scalar to a Python scalar.

    For instance, `3.0`, `np.float32(3.0)`, and `np.float64(3.0)` all
    map to `3.0`.

    Args:
      value: A Python scalar (`int`, `float`, `str`, or `bool`) or
        rank-0 `numpy` equivalent (e.g., `np.int64`, `np.float32`).

    Returns:
      A Python scalar equivalent to `value`.
    """
    if isinstance(value, np.generic):
        return value.item()
    else:
        return value