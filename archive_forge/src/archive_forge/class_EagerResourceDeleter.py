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
class EagerResourceDeleter:
    """An object which cleans up a resource handle.

  An alternative to defining a __del__ method on an object. The intended use is
  that ResourceVariables or other objects with resource handles will maintain a
  single reference to this object. When the parent object is collected, this
  object will be too. Even if the parent object is part of a reference cycle,
  the cycle will be collectable.
  """
    __slots__ = ['_handle', '_handle_device', '_context']

    def __init__(self, handle, handle_device):
        if not isinstance(handle, tensor_module.Tensor):
            raise ValueError(f'Passed handle={handle} to EagerResourceDeleter. Was expecting the handle to be a `tf.Tensor`.')
        self._handle = handle
        self._handle_device = handle_device
        self._context = context.context()

    def __del__(self):
        try:
            if isinstance(self._handle, ops.EagerTensor) and self._handle.is_packed:
                return
            with context.eager_mode():
                with ops.device(self._handle_device):
                    gen_resource_variable_ops.destroy_resource_op(self._handle, ignore_lookup_error=True)
        except TypeError:
            pass
        except AttributeError:
            pass