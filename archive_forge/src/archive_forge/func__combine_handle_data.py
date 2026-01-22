import contextlib
import functools
import weakref
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import variable_pb2
from tensorflow.core.function import trace_type
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.checkpoint import tensor_callable
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.compat import compat as forward_compat
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.eager import tape
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_module
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.gen_resource_variable_ops import *
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
def _combine_handle_data(handle, initial_value):
    """Concats HandleData from tensors `handle` and `initial_value`.

  Args:
    handle: A `Tensor` of dtype `resource`.
    initial_value: A `Tensor`.

  Returns:
    A `CppShapeInferenceResult.HandleData`.  If `initial_value` has dtype
    `variant`, the `HandleData` contains the concatenation of the shape_and_type
    from both `handle` and `initial_value`.

  Raises:
    RuntimeError: If handle, which was returned by VarHandleOp, either has
      no handle data, or its len(handle_data.shape_and_type) != 1.
  """
    assert handle.dtype == dtypes.resource
    variable_handle_data = get_eager_safe_handle_data(handle)
    if initial_value.dtype != dtypes.variant:
        return variable_handle_data
    extra_handle_data = get_eager_safe_handle_data(initial_value)
    if extra_handle_data is not None and extra_handle_data.is_set:
        if variable_handle_data is None or not variable_handle_data.is_set or len(variable_handle_data.shape_and_type) != 1:
            raise RuntimeError(f"Expected VarHandleOp to return a length==1 shape_and_type, but saw: '{variable_handle_data}'")
        variable_handle_data.shape_and_type.extend(extra_handle_data.shape_and_type)
    return variable_handle_data