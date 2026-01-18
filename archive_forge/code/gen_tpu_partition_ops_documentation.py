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
An op that demultiplexes a tensor to be sharded by XLA to a list of partitioned

  outputs outside the XLA computation. Supports ND sharding.

  Args:
    inputs: A `Tensor`.
      A tensor which represents the full shape of partitioned tensors.
    num_splits: An `int` that is `>= 1`.
    partition_dims: A list of `ints`.
      A list of integers describing how each dimension is partitioned. Emptiness
      indicates the inputs are replicated.
    name: A name for the operation (optional).

  Returns:
    A list of `num_splits` `Tensor` objects with the same type as `inputs`.
  