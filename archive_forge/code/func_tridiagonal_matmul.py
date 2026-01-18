import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('linalg.tridiagonal_matmul')
@dispatch.add_dispatch_support
def tridiagonal_matmul(diagonals, rhs, diagonals_format='compact', name=None):
    """Multiplies tridiagonal matrix by matrix.

  `diagonals` is representation of 3-diagonal NxN matrix, which depends on
  `diagonals_format`.

  In `matrix` format, `diagonals` must be a tensor of shape `[..., M, M]`, with
  two inner-most dimensions representing the square tridiagonal matrices.
  Elements outside of the three diagonals will be ignored.

  If `sequence` format, `diagonals` is list or tuple of three tensors:
  `[superdiag, maindiag, subdiag]`, each having shape [..., M]. Last element
  of `superdiag` first element of `subdiag` are ignored.

  In `compact` format the three diagonals are brought together into one tensor
  of shape `[..., 3, M]`, with last two dimensions containing superdiagonals,
  diagonals, and subdiagonals, in order. Similarly to `sequence` format,
  elements `diagonals[..., 0, M-1]` and `diagonals[..., 2, 0]` are ignored.

  The `sequence` format is recommended as the one with the best performance.

  `rhs` is matrix to the right of multiplication. It has shape `[..., M, N]`.

  Example:

  ```python
  superdiag = tf.constant([-1, -1, 0], dtype=tf.float64)
  maindiag = tf.constant([2, 2, 2], dtype=tf.float64)
  subdiag = tf.constant([0, -1, -1], dtype=tf.float64)
  diagonals = [superdiag, maindiag, subdiag]
  rhs = tf.constant([[1, 1], [1, 1], [1, 1]], dtype=tf.float64)
  x = tf.linalg.tridiagonal_matmul(diagonals, rhs, diagonals_format='sequence')
  ```

  Args:
    diagonals: A `Tensor` or tuple of `Tensor`s describing left-hand sides. The
      shape depends of `diagonals_format`, see description above. Must be
      `float32`, `float64`, `complex64`, or `complex128`.
    rhs: A `Tensor` of shape [..., M, N] and with the same dtype as `diagonals`.
    diagonals_format: one of `sequence`, or `compact`. Default is `compact`.
    name:  A name to give this `Op` (optional).

  Returns:
    A `Tensor` of shape [..., M, N] containing the result of multiplication.

  Raises:
    ValueError: An unsupported type is provided as input, or when the input
    tensors have incorrect shapes.
  """
    if diagonals_format == 'compact':
        superdiag = diagonals[..., 0, :]
        maindiag = diagonals[..., 1, :]
        subdiag = diagonals[..., 2, :]
    elif diagonals_format == 'sequence':
        superdiag, maindiag, subdiag = diagonals
    elif diagonals_format == 'matrix':
        m1 = tensor_shape.dimension_value(diagonals.shape[-1])
        m2 = tensor_shape.dimension_value(diagonals.shape[-2])
        if m1 and m2 and (m1 != m2):
            raise ValueError('Expected last two dimensions of diagonals to be same, got {} and {}'.format(m1, m2))
        diags = array_ops.matrix_diag_part(diagonals, k=(-1, 1), padding_value=0.0, align='LEFT_RIGHT')
        superdiag = diags[..., 0, :]
        maindiag = diags[..., 1, :]
        subdiag = diags[..., 2, :]
    else:
        raise ValueError('Unrecognized diagonals_format: %s' % diagonals_format)
    superdiag = array_ops.expand_dims(superdiag, -2)
    maindiag = array_ops.expand_dims(maindiag, -2)
    subdiag = array_ops.expand_dims(subdiag, -2)
    return linalg_ops.tridiagonal_mat_mul(superdiag, maindiag, subdiag, rhs, name)