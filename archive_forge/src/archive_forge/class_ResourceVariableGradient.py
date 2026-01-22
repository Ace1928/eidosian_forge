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
class ResourceVariableGradient(composite_tensor_gradient.CompositeTensorGradient):
    """CompositeTensorGradient protocol for ResourceVariable."""

    def get_gradient_components(self, value):
        """Returns the components of `value` that should be included in gradients.

    For a ResourceVariable, its gradient component is its handle tensor.
    For now, we return the ResourceVariable because the gradient infrastructure
    has special logics to handle ResourceVariables. We should remove those
    special logics and return the handle tensor.

    Args:
      value: A `ResourceVariable`.

    Returns:
      `value` itself.
    """
        return value

    def replace_gradient_components(self, value, component_grads):
        """Replaces the gradient components in `value` with `component_grads`.

    The gradient of a ResourceVariable is either None or a Tensor. So we don't
    need `value`'s TypeSpec or non-gradient components in this method.

    Args:
      value: A `ResourceVariable` with its gradient components compatible with
        `component_grads`.
      component_grads: A `Tensor` or None as the gradient result.

    Returns:
      The `component_grads`, which is either a `Tensor` or None.
    """
        return component_grads