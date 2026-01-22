import copy
import math
from typing import Sequence
import weakref
import numpy as np
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices as indexed_slices_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.saved_model import save_context
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _decompose_slice_spec(self, slice_spec):
    """Decompose a global slice_spec into a list of per-variable slice_spec.

    `ShardedVariable` only supports first dimension partitioning, thus
    `slice_spec` must be for first dimension.

    Args:
      slice_spec: A python `slice` object that specifies the global slicing.

    Returns:
      A list of python `slice` objects or None specifying the local slicing for
      each component variable. None means no slicing.

    For example, given component variables:
      v0 = [0, 1, 2]
      v1 = [3, 4, 5]
      v2 = [6, 7, 8, 9]

    If `slice_spec` is slice(start=None, stop=None, step=None), we will have:
      v0[returned[0]] = [0, 1, 2]
      v1[returned[1]] = [3, 4, 5]
      v2[returned[2]] = [6, 7, 8, 9]
    If `slice_spec` is slice(start=2, stop=8, step=3), we will have:
      v0[returned[0]] = [2]
      v1[returned[1]] = [5]
      returned[2] == None
    If `slice_spec` is slice(start=9, stop=3, step=-2), we will have:
      returned[0] == None
      v1[returned[1]] = [5]
      v2[returned[2]] = [9, 7]
    """
    if isinstance(slice_spec.start, tensor_lib.Tensor) or isinstance(slice_spec.stop, tensor_lib.Tensor) or isinstance(slice_spec.step, tensor_lib.Tensor):
        raise TypeError('ShardedVariable: using Tensor in slice_spec is not allowed. Please file a feature request with the TensorFlow team.')
    result = []
    slice_step = slice_spec.step if slice_spec.step is not None else 1
    if slice_step == 0:
        raise ValueError('slice step cannot be zero')
    slice_start = slice_spec.start
    if slice_start is None:
        slice_start = 0 if slice_step > 0 else self._shape[0] - 1
    elif slice_start < 0:
        slice_start += self._shape[0]
    slice_end = slice_spec.stop
    if slice_end is None:
        slice_end = self._shape[0] if slice_step > 0 else -1
    elif slice_end < 0:
        slice_end += self._shape[0]
    cur = slice_start
    if slice_step > 0:
        for i in range(len(self._var_offsets)):
            var_start = self._var_offsets[i][0]
            var_end = self._var_offsets[i + 1][0] if i < len(self._var_offsets) - 1 else self._shape[0]
            if cur < var_start:
                cur += slice_step * int(math.ceil((var_start - cur) / slice_step))
            if cur >= var_end or cur >= slice_end:
                result.append(None)
            else:
                start = cur - var_start
                end = min(slice_end, var_end) - var_start
                result.append(slice(start, end, slice_step))
    else:
        for i in range(len(self._var_offsets) - 1, -1, -1):
            var_start = self._var_offsets[i][0]
            var_end = self._var_offsets[i + 1][0] if i < len(self._var_offsets) - 1 else self._shape[0]
            if cur >= var_end:
                cur += slice_step * int(math.ceil((var_end - cur - 1) / slice_step))
            if cur < var_start or cur <= slice_end:
                result.append(None)
            else:
                start = cur - var_start
                if slice_end >= var_start:
                    end = slice_end - var_start
                else:
                    end = None
                result.append(slice(start, end, slice_step))
        result.reverse()
    return result