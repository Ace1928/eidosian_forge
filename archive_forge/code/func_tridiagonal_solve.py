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
@tf_export('linalg.tridiagonal_solve')
@dispatch.add_dispatch_support
def tridiagonal_solve(diagonals, rhs, diagonals_format='compact', transpose_rhs=False, conjugate_rhs=False, name=None, partial_pivoting=True, perturb_singular=False):
    """Solves tridiagonal systems of equations.

  The input can be supplied in various formats: `matrix`, `sequence` and
  `compact`, specified by the `diagonals_format` arg.

  In `matrix` format, `diagonals` must be a tensor of shape `[..., M, M]`, with
  two inner-most dimensions representing the square tridiagonal matrices.
  Elements outside of the three diagonals will be ignored.

  In `sequence` format, `diagonals` are supplied as a tuple or list of three
  tensors of shapes `[..., N]`, `[..., M]`, `[..., N]` representing
  superdiagonals, diagonals, and subdiagonals, respectively. `N` can be either
  `M-1` or `M`; in the latter case, the last element of superdiagonal and the
  first element of subdiagonal will be ignored.

  In `compact` format the three diagonals are brought together into one tensor
  of shape `[..., 3, M]`, with last two dimensions containing superdiagonals,
  diagonals, and subdiagonals, in order. Similarly to `sequence` format,
  elements `diagonals[..., 0, M-1]` and `diagonals[..., 2, 0]` are ignored.

  The `compact` format is recommended as the one with best performance. In case
  you need to cast a tensor into a compact format manually, use `tf.gather_nd`.
  An example for a tensor of shape [m, m]:

  ```python
  rhs = tf.constant([...])
  matrix = tf.constant([[...]])
  m = matrix.shape[0]
  dummy_idx = [0, 0]  # An arbitrary element to use as a dummy
  indices = [[[i, i + 1] for i in range(m - 1)] + [dummy_idx],  # Superdiagonal
           [[i, i] for i in range(m)],                          # Diagonal
           [dummy_idx] + [[i + 1, i] for i in range(m - 1)]]    # Subdiagonal
  diagonals=tf.gather_nd(matrix, indices)
  x = tf.linalg.tridiagonal_solve(diagonals, rhs)
  ```

  Regardless of the `diagonals_format`, `rhs` is a tensor of shape `[..., M]` or
  `[..., M, K]`. The latter allows to simultaneously solve K systems with the
  same left-hand sides and K different right-hand sides. If `transpose_rhs`
  is set to `True` the expected shape is `[..., M]` or `[..., K, M]`.

  The batch dimensions, denoted as `...`, must be the same in `diagonals` and
  `rhs`.

  The output is a tensor of the same shape as `rhs`: either `[..., M]` or
  `[..., M, K]`.

  The op isn't guaranteed to raise an error if the input matrix is not
  invertible. `tf.debugging.check_numerics` can be applied to the output to
  detect invertibility problems.

  **Note**: with large batch sizes, the computation on the GPU may be slow, if
  either `partial_pivoting=True` or there are multiple right-hand sides
  (`K > 1`). If this issue arises, consider if it's possible to disable pivoting
  and have `K = 1`, or, alternatively, consider using CPU.

  On CPU, solution is computed via Gaussian elimination with or without partial
  pivoting, depending on `partial_pivoting` parameter. On GPU, Nvidia's cuSPARSE
  library is used: https://docs.nvidia.com/cuda/cusparse/index.html#gtsv

  Args:
    diagonals: A `Tensor` or tuple of `Tensor`s describing left-hand sides. The
      shape depends of `diagonals_format`, see description above. Must be
      `float32`, `float64`, `complex64`, or `complex128`.
    rhs: A `Tensor` of shape [..., M] or [..., M, K] and with the same dtype as
      `diagonals`. Note that if the shape of `rhs` and/or `diags` isn't known
      statically, `rhs` will be treated as a matrix rather than a vector.
    diagonals_format: one of `matrix`, `sequence`, or `compact`. Default is
      `compact`.
    transpose_rhs: If `True`, `rhs` is transposed before solving (has no effect
      if the shape of rhs is [..., M]).
    conjugate_rhs: If `True`, `rhs` is conjugated before solving.
    name:  A name to give this `Op` (optional).
    partial_pivoting: whether to perform partial pivoting. `True` by default.
      Partial pivoting makes the procedure more stable, but slower. Partial
      pivoting is unnecessary in some cases, including diagonally dominant and
      symmetric positive definite matrices (see e.g. theorem 9.12 in [1]).
    perturb_singular: whether to perturb singular matrices to return a finite
      result. `False` by default. If true, solutions to systems involving
      a singular matrix will be computed by perturbing near-zero pivots in
      the partially pivoted LU decomposition. Specifically, tiny pivots are
      perturbed by an amount of order `eps * max_{ij} |U(i,j)|` to avoid
      overflow. Here `U` is the upper triangular part of the LU decomposition,
      and `eps` is the machine precision. This is useful for solving
      numerically singular systems when computing eigenvectors by inverse
      iteration.
      If `partial_pivoting` is `False`, `perturb_singular` must be `False` as
      well.

  Returns:
    A `Tensor` of shape [..., M] or [..., M, K] containing the solutions.
    If the input matrix is singular, the result is undefined.

  Raises:
    ValueError: Is raised if any of the following conditions hold:
      1. An unsupported type is provided as input,
      2. the input tensors have incorrect shapes,
      3. `perturb_singular` is `True` but `partial_pivoting` is not.
    UnimplementedError: Whenever `partial_pivoting` is true and the backend is
      XLA, or whenever `perturb_singular` is true and the backend is
      XLA or GPU.

  [1] Nicholas J. Higham (2002). Accuracy and Stability of Numerical Algorithms:
  Second Edition. SIAM. p. 175. ISBN 978-0-89871-802-7.

  """
    if perturb_singular and (not partial_pivoting):
        raise ValueError('partial_pivoting must be True if perturb_singular is.')
    if diagonals_format == 'compact':
        return _tridiagonal_solve_compact_format(diagonals, rhs, transpose_rhs, conjugate_rhs, partial_pivoting, perturb_singular, name)
    if diagonals_format == 'sequence':
        if not isinstance(diagonals, (tuple, list)) or len(diagonals) != 3:
            raise ValueError('Expected diagonals to be a sequence of length 3.')
        superdiag, maindiag, subdiag = diagonals
        if not subdiag.shape[:-1].is_compatible_with(maindiag.shape[:-1]) or not superdiag.shape[:-1].is_compatible_with(maindiag.shape[:-1]):
            raise ValueError('Tensors representing the three diagonals must have the same shape,except for the last dimension, got {}, {}, {}'.format(subdiag.shape, maindiag.shape, superdiag.shape))
        m = tensor_shape.dimension_value(maindiag.shape[-1])

        def pad_if_necessary(t, name, last_dim_padding):
            n = tensor_shape.dimension_value(t.shape[-1])
            if not n or n == m:
                return t
            if n == m - 1:
                paddings = [[0, 0] for _ in range(len(t.shape) - 1)] + [last_dim_padding]
                return array_ops.pad(t, paddings)
            raise ValueError('Expected {} to be have length {} or {}, got {}.'.format(name, m, m - 1, n))
        subdiag = pad_if_necessary(subdiag, 'subdiagonal', [1, 0])
        superdiag = pad_if_necessary(superdiag, 'superdiagonal', [0, 1])
        diagonals = array_ops_stack.stack((superdiag, maindiag, subdiag), axis=-2)
        return _tridiagonal_solve_compact_format(diagonals, rhs, transpose_rhs, conjugate_rhs, partial_pivoting, perturb_singular, name)
    if diagonals_format == 'matrix':
        m1 = tensor_shape.dimension_value(diagonals.shape[-1])
        m2 = tensor_shape.dimension_value(diagonals.shape[-2])
        if m1 and m2 and (m1 != m2):
            raise ValueError('Expected last two dimensions of diagonals to be same, got {} and {}'.format(m1, m2))
        m = m1 or m2
        diagonals = array_ops.matrix_diag_part(diagonals, k=(-1, 1), padding_value=0.0, align='LEFT_RIGHT')
        return _tridiagonal_solve_compact_format(diagonals, rhs, transpose_rhs, conjugate_rhs, partial_pivoting, perturb_singular, name)
    raise ValueError('Unrecognized diagonals_format: {}'.format(diagonals_format))