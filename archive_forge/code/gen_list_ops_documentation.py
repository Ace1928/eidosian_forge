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
Stacks all tensors in the list.

  Requires that all tensors have the same shape.

  input_handle: the input list
  tensor: the gathered result
  num_elements: optional. If not -1, the number of elements in the list.

  Args:
    input_handle: A `Tensor` of type `variant`.
    element_shape: A `Tensor` of type `int32`.
    element_dtype: A `tf.DType`.
    num_elements: An optional `int`. Defaults to `-1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `element_dtype`.
  