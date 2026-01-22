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
def _popitem(self, indices=None, name=None):
    """If the staging area is ordered, the (key, value) with the smallest key will be returned.

    Otherwise, a random (key, value) will be returned.
    If the staging area is empty when this operation executes,
    it will block until there is an element to dequeue.

    Args:
        key: Key associated with the required data
        indices: Partial list of tensors to retrieve (optional).
                A list of integer or string indices.
                String indices are only valid if the Staging Area
                has names associated with it.
        name: A name for the operation (optional)

    Returns:
        The created op
    """
    if name is None:
        name = '%s_get_nokey' % self._name
    indices, dtypes = self._get_indices_and_dtypes(indices)
    with ops.colocate_with(self._coloc_op):
        key, result = self._popitem_fn(shared_name=self._name, indices=indices, dtypes=dtypes, name=name, capacity=self._capacity, memory_limit=self._memory_limit)
    key = self._create_device_transfers(key)[0]
    result = self._get_return_value(result, indices)
    return (key, result)