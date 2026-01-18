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
def write_object_proto_for_resource_variable(resource_variable, proto, options, enforce_naming=True):
    """Writes additional information of the variable into the SavedObject proto.

  This allows users to define a `hook` to provide extra information of the
  variable to the SavedObject.

  For example, DistributedVariable class would fill in components in the
  distributed context.

  Args:
    resource_variable: A `ResourceVariable` or `DistributedValue` that has the
      information to be saved into the proto.
    proto: `SavedObject` proto to update.
    options: A `SaveOption` instance that configures save behavior.
    enforce_naming: A bool determining whether to check that names end in the
      expected string ':0'
  """
    proto.variable.SetInParent()
    if enforce_naming and (not resource_variable.name.endswith(':0')):
        raise ValueError(f"Cowardly refusing to save variable {resource_variable.name} because of unexpected suffix in the name (expected ':0')which won't be restored.")
    proto.variable.name = tensor_module.get_op_name(resource_variable.name)
    proto.variable.trainable = resource_variable.trainable
    proto.variable.dtype = resource_variable.dtype.as_datatype_enum
    proto.variable.synchronization = resource_variable.synchronization.value
    proto.variable.aggregation = resource_variable.aggregation.value
    proto.variable.shape.CopyFrom(resource_variable.shape.as_proto())
    if options.experimental_variable_policy._save_variable_devices():
        if hasattr(resource_variable, 'device'):
            proto.variable.device = resource_variable.device