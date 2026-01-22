import re
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import dynamic_ragged_shape
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
@classmethod
def _from_pylist_of_value(cls, pyval, typespec, path_so_far):
    """Converts python list `pyval` to a Tensor or RaggedTensor with rank>1."""
    if typespec is None:
        try:
            return ragged_factory_ops.constant(pyval)
        except Exception as exc:
            raise ValueError('Error parsing path %r' % (path_so_far,)) from exc
    elif isinstance(typespec, tensor.TensorSpec):
        try:
            result = constant_op.constant(pyval, typespec.dtype)
        except Exception as exc:
            raise ValueError('Error parsing path %r' % (path_so_far,)) from exc
        if not typespec.shape.is_compatible_with(result.shape):
            raise ValueError('Value at %r does not match typespec: %r vs %r' % (path_so_far, typespec, pyval))
        return result
    elif isinstance(typespec, ragged_tensor.RaggedTensorSpec):
        try:
            return ragged_factory_ops.constant(pyval, dtype=typespec._dtype, ragged_rank=typespec._ragged_rank, row_splits_dtype=typespec._row_splits_dtype, inner_shape=typespec._shape[typespec._ragged_rank + 1:])
        except Exception as exc:
            raise ValueError('Error parsing path %r' % (path_so_far,)) from exc
    elif isinstance(typespec, StructuredTensor.Spec):
        empty_rank = _pyval_empty_list_depth(pyval)
        if empty_rank is None:
            raise ValueError('Value at %r does not match typespec: %r vs %r' % (path_so_far, typespec, pyval))
        else:
            return cls._from_pylist_of_dict(pyval, set(), empty_rank, typespec, path_so_far)
    else:
        raise ValueError('Value at %r does not match typespec: %r vs %r' % (path_so_far, typespec, pyval))