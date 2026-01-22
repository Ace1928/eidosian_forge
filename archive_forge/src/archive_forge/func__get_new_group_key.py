import copy
import threading
from typing import Callable, List, Optional, Union
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import values as value_lib
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import core
def _get_new_group_key(self, devices):
    """Returns a new group key.

    The caller should store and reuse the same group key for the same set of
    devices. Calling this method always returns a new group key.

    This method is not thread-safe.

    Args:
      devices: a list of canonical device strings in a collective group.

    Returns:
      a new group key.
    """
    new_key = self._group_key
    self._group_key += 1
    self._instance_key_table[new_key] = {}
    for device in devices:
        self._instance_key_table[new_key][device] = INSTANCE_KEY_START_NUMBER
    return new_key