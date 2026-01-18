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
Outputs the position of `value` in a permutation of [0, ..., max_index].

  Output values are a bijection of the `index` for any combination and `seed` and `max_index`.

  If multiple inputs are vectors (matrix in case of seed) then the size of the
  first dimension must match.

  The outputs are deterministic.

  Args:
    index: A `Tensor`. Must be one of the following types: `int32`, `uint32`, `int64`, `uint64`.
      A scalar tensor or a vector of dtype `dtype`. The index (or indices) to be shuffled. Must be within [0, max_index].
    seed: A `Tensor`. Must be one of the following types: `int32`, `uint32`, `int64`, `uint64`.
      A tensor of dtype `Tseed` and shape [3] or [n, 3]. The random seed.
    max_index: A `Tensor`. Must have the same type as `index`.
      A scalar tensor or vector of dtype `dtype`. The upper bound(s) of the interval (inclusive).
    rounds: An optional `int`. Defaults to `4`.
      The number of rounds to use the in block cipher.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `index`.
  