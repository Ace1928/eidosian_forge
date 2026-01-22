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
def _take_many_sparse_from_tensors_map(sparse_map_op, sparse_handles, rank=None, name=None):
    """Read `SparseTensors` from a `SparseTensorsMap` and concatenate them.

  The input `sparse_handles` must be a string matrix of shape `[N, 1]` where
  `N` is the minibatch size and the rows correspond to packed outputs of
  `add_sparse_to_tensors_map`.  The ranks of the original `SparseTensor` objects
  must all match.  When the final `SparseTensor` is created, it has rank one
  higher than the ranks of the incoming `SparseTensor` objects (they have been
  concatenated along a new row dimension).

  The output `SparseTensor` object's shape values for all dimensions but the
  first are the max across the input `SparseTensor` objects' shape values
  for the corresponding dimensions.  Its first shape value is `N`, the minibatch
  size.

  The input `SparseTensor` objects' indices are assumed ordered in
  standard lexicographic order.  If this is not the case, after this
  step run `sparse.reorder` to restore index ordering.

  For example, if the serialized input is a `[2, 3]` matrix representing two
  original `SparseTensor` objects:

      index = [ 0]
              [10]
              [20]
      values = [1, 2, 3]
      shape = [50]

  and

      index = [ 2]
              [10]
      values = [4, 5]
      shape = [30]

  then the final deserialized `SparseTensor` will be:

      index = [0  0]
              [0 10]
              [0 20]
              [1  2]
              [1 10]
      values = [1, 2, 3, 4, 5]
      shape = [2 50]

  Args:
    sparse_map_op: The `Operation` that created the original handles.
      Usually this is, e.g., `add_sparse_to_tensors_map(...).op`.
    sparse_handles: 2-D `Tensor` of type `string` of shape `[N, 1]`.
      The serialized and packed `SparseTensor` objects.
    rank: (optional) Python int, the rank of the `SparseTensor` objects.
    name: A name prefix for the returned tensors (optional)

  Returns:
    A `SparseTensor` representing the deserialized `SparseTensor`s,
    concatenated along the `SparseTensor`s' first dimension.

    All of the serialized `SparseTensor`s must have had the same rank and type.
  """
    if not isinstance(sparse_map_op, ops.Operation):
        raise TypeError('sparse_map_op be an Operation')
    if sparse_map_op.type not in ('AddSparseToTensorsMap', 'AddManySparseToTensorsMap'):
        raise TypeError('sparse_map_op must be one of AddSparseToTensorsMap or AddSparseToTensorsMap. Instead, found `%s`.' % sparse_map_op.type)
    with ops.colocate_with(sparse_map_op):
        shared_name = sparse_map_op.get_attr('shared_name') or sparse_map_op.name
        output_indices, output_values, output_shape = gen_sparse_ops.take_many_sparse_from_tensors_map(sparse_handles, dtype=sparse_map_op.get_attr('T'), container=sparse_map_op.get_attr('container'), shared_name=shared_name, name=name)
    output_indices.set_shape([None, rank])
    output_shape.set_shape([rank])
    return sparse_tensor.SparseTensor(output_indices, output_values, output_shape)