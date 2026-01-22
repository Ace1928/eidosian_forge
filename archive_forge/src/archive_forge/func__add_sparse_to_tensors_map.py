import numbers
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import bincount_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_count_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops.gen_sparse_ops import *
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import get_canonical_name_for_symbol
from tensorflow.python.util.tf_export import tf_export
def _add_sparse_to_tensors_map(sp_input, container=None, shared_name=None, name=None):
    """Add a `SparseTensor` to a `SparseTensorsMap` and return its handle.

  Args:
    sp_input: The input `SparseTensor`.
    container: The container for the underlying `SparseTensorsMap` (optional).
    shared_name: The shared name for the underlying `SparseTensorsMap`
      (optional, defaults to the name of the newly created op).
    name: A name prefix for the returned tensors (optional).

  Returns:
    A string 1-vector (1D `Tensor`), with the single element representing the
    a unique handle to a `SparseTensor` stored by the `SparseTensorMap`
    underlying this op.

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
    sp_input = _convert_to_sparse_tensor(sp_input)
    return gen_sparse_ops.add_sparse_to_tensors_map(sp_input.indices, sp_input.values, sp_input.dense_shape, container=container, shared_name=shared_name, name=name)