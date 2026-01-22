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
def _replace_sparse_with_values(value, sparse_list):
    """Replace `SparseTensor`s with their values in `value`

  Each `SparseTensor` in `value` is replaced by its `values` tensor, and
  collects all `SparseTensor`s in `sparse_list`.

  Args:
    value: A structure of `Tensor`s and `SparseTensor`s
    sparse_list: A list. Output parameter that collects all `SparseTensor`s in
      `value`.

  Returns:
    `value` with each SparseTensor replaced by its `.value` attribute.
  """
    flat_vals = nest.flatten(value, expand_composites=False)
    new_vals = []
    for v in flat_vals:
        if isinstance(v, sparse_tensor.SparseTensor):
            sparse_list.append(v)
            new_vals.append(v.values)
        else:
            new_vals.append(v)
    return nest.pack_sequence_as(value, new_vals, expand_composites=False)