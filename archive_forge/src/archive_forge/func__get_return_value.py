import functools
import hashlib
import threading
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.gen_data_flow_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def _get_return_value(self, tensors, indices):
    """Return the value to return from a get op.

    If the staging area has names, return a dictionary with the
    names as keys.  Otherwise return either a single tensor
    or a list of tensors depending on the length of `tensors`.

    Args:
      tensors: List of tensors from the get op.
      indices: Indices of associated names and shapes

    Returns:
      A single tensor, a list of tensors, or a dictionary
      of tensors.
    """
    tensors = self._create_device_transfers(tensors)
    for output, i in zip(tensors, indices):
        output.set_shape(self._shapes[i])
    if self._names:
        return {self._names[i]: t for t, i in zip(tensors, indices)}
    return tensors