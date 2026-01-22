import abc
import contextlib
import numpy as np
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator_algebra
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.ops.linalg import slicing
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def _extract_type_spec_recursively(value):
    """Return (collection of) `TypeSpec`(s) for `value` if it includes `Tensor`s.

  If `value` is a `Tensor` or `CompositeTensor`, return its `TypeSpec`. If
  `value` is a collection containing `Tensor` values, recursively supplant them
  with their respective `TypeSpec`s in a collection of parallel stucture.

  If `value` is none of the above, return it unchanged.

  Args:
    value: a Python `object` to (possibly) turn into a (collection of)
    `tf.TypeSpec`(s).

  Returns:
    spec: the `TypeSpec` or collection of `TypeSpec`s corresponding to `value`
    or `value`, if no `Tensor`s are found.
  """
    if isinstance(value, composite_tensor.CompositeTensor):
        return value._type_spec
    if isinstance(value, variables.Variable):
        return resource_variable_ops.VariableSpec(value.shape, dtype=value.dtype, trainable=value.trainable)
    if tensor_util.is_tensor(value):
        return tensor_spec.TensorSpec(value.shape, value.dtype)
    if isinstance(value, list):
        return list((_extract_type_spec_recursively(v) for v in value))
    if isinstance(value, data_structures.TrackableDataStructure):
        return _extract_type_spec_recursively(value.__wrapped__)
    if isinstance(value, tuple):
        return type(value)((_extract_type_spec_recursively(x) for x in value))
    if isinstance(value, dict):
        return type(value)(((k, _extract_type_spec_recursively(v)) for k, v in value.items()))
    return value