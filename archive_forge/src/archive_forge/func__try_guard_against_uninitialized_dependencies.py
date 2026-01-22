import abc
import enum
import functools
import itertools
import os
from tensorflow.core.framework import variable_pb2
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_should_use
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export
def _try_guard_against_uninitialized_dependencies(name, initial_value):
    """Attempt to guard against dependencies on uninitialized variables.

  Replace references to variables in `initial_value` with references to the
  variable's initialized values. The initialized values are essentially
  conditional TensorFlow graphs that return a variable's value if it is
  initialized or its `initial_value` if it hasn't been initialized. This
  replacement is done on a best effort basis:

  - If the `initial_value` graph contains cycles, we don't do any
    replacements for that graph.
  - If the variables that `initial_value` depends on are not present in the
    `GLOBAL_VARIABLES` or `LOCAL_VARIABLES` we don't replace them.

  In these cases, it is up to the caller to ensure that the `initial_value`
  graph uses initialized variables or that they guard access to variables
  using their `initialized_value` method.

  Args:
    name: Variable name.
    initial_value: `Tensor`. The initial value.

  Returns:
    A `Tensor` suitable to initialize a variable.
  Raises:
    TypeError: If `initial_value` is not a `Tensor`.
  """
    if not isinstance(initial_value, tensor_lib.Tensor):
        raise TypeError('initial_value needs to be a Tensor: %s' % initial_value)
    if _has_cycle(initial_value.op, state={}):
        return initial_value
    return _safe_initial_value_from_tensor(name, initial_value, op_cache={})