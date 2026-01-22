import functools
import typing
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_ragged_math_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _matmul_2d(a, b, **kwargs):
    """Multiplies potentially ragged 2D tensors.

  Args:
    a: A 2D Tensor or RaggedTensor with `shape=[I, J]`
    b: A 2D Tensor or RaggedTensor with `shape=[J, K]`
    **kwargs: Additional arguments for `tf.matmul` (e.g. transpose_a).

  Returns:
    A 2D Tensor with `shape=[I, K]`.
  """
    ragged_err = 'The matrices in `a` and `b` may not be ragged in their innermost dimension.'
    checks = []
    if isinstance(a, ragged_tensor.RaggedTensor):
        original_size = array_ops.size(a.flat_values)
        a = a.to_tensor()
        checks.append(check_ops.assert_equal(original_size, array_ops.size(a), message=ragged_err))
    if isinstance(b, ragged_tensor.RaggedTensor):
        original_size = array_ops.size(b.flat_values)
        b = b.to_tensor()
        checks.append(check_ops.assert_equal(original_size, array_ops.size(b), message=ragged_err))
    with ops.control_dependencies(checks):
        return math_ops.matmul(a, b, **kwargs)