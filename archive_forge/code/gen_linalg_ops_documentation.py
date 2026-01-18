import collections
from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.security.fuzzing.py import annotation_types as _atypes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export
from typing import TypeVar, List
Solves tridiagonal systems of equations.

    Solves tridiagonal systems of equations.
    Supports batch dimensions and multiple right-hand sides per each left-hand
    side.
    On CPU, solution is computed via Gaussian elimination with or without partial
    pivoting, depending on `partial_pivoting` attribute. On GPU, Nvidia's cuSPARSE
    library is used: https://docs.nvidia.com/cuda/cusparse/index.html#gtsv
    Partial pivoting is not yet supported by XLA backends.

  Args:
    diagonals: A `Tensor`. Must be one of the following types: `float64`, `float32`, `complex64`, `complex128`.
      Tensor of shape `[..., 3, M]` whose innermost 2 dimensions represent the
      tridiagonal matrices with three rows being the superdiagonal, diagonals, and
      subdiagonals, in order. The last element of the superdiagonal and the first
      element of the subdiagonal is ignored.
    rhs: A `Tensor`. Must have the same type as `diagonals`.
      Tensor of shape `[..., M, K]`, representing K right-hand sides per each
      left-hand side.
    partial_pivoting: An optional `bool`. Defaults to `True`.
      Whether to apply partial pivoting. Partial pivoting makes the procedure more
      stable, but slower.
    perturb_singular: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `diagonals`.
  