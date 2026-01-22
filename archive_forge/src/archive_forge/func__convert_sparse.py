import collections
from functools import partial
import string
import sys
import traceback
import numpy as np
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.core.framework import full_type_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import execute
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_switch_case
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import gen_list_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_optional_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import gen_spectral_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import flags
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
def _convert_sparse(self, y):
    """Returns the converted value corresponding to SparseTensor y.

    For SparseTensors, instead of stacking the component tensors separately,
    resulting in component tensors with shapes (N, m, rank), (N, m), and (N,
    rank) respectively for indices, values, and dense_shape (where N is the loop
    length and m is the number of sparse tensor values per loop iter), we want
    to logically stack the SparseTensors, to create a SparseTensor whose
    components are size (N * m, rank + 1), (N * m, ), and (rank + 1,)
    respectively.

    Here, we try to get the conversion of each component tensor.
    If the tensors are stacked via a sparse conversion, return the resulting
    SparseTensor composed of the converted components. Otherwise, the component
    tensors are either unstacked or stacked naively. In the latter case, we
    unstack the component tensors to reform loop_len SparseTensor elements,
    then correctly batch them.

    The unstacked tensors must have the same rank. Each dimension of each
    SparseTensor will expand to be the largest among all SparseTensor elements
    for that dimension. For example, if there are N SparseTensors of rank 3
    being stacked, with N dense shapes, where the i_th shape is (x_i, y_i, z_i),
    the new dense shape will be (N, max_i(x_i), max_i(y_i), max_i(z_i)).

    Args:
      y: A tf.sparse.SparseTensor.

    Returns:
      A tf.sparse.SparseTensor that is the converted value corresponding to y.
    """
    outputs = [self._convert_helper(t) for t in (y.indices, y.values, y.dense_shape)]
    assert all((isinstance(o, WrappedTensor) for o in outputs))
    if all((w.is_sparse_stacked for w in outputs)):
        return sparse_tensor.SparseTensor(*[w.t for w in outputs])
    assert not any((w.is_sparse_stacked for w in outputs)), 'Error converting SparseTensor. All components should be logically stacked, or none.'
    return self._restack_sparse_tensor_logically(*[self._unwrap_or_tile(w) for w in outputs])