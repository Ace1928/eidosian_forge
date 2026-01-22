from typing import Optional, Type
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.function import trace_type
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.eager import record
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import stack
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import handle_data_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import core as core_tf_types
from tensorflow.python.types import internal
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
def _graph_placeholder(self, graph, name=None):
    """Graph-only version of tf.compat.v1.placeholder(), for internal use only."""
    dtype = self.dtype.base_dtype
    shape = self.shape
    dtype_value = attr_value_pb2.AttrValue(type=dtype.as_datatype_enum)
    if isinstance(shape, (list, tuple)):
        shape = tensor_shape.TensorShape(shape)
    shape = attr_value_pb2.AttrValue(shape=shape.as_proto())
    attrs = {'dtype': dtype_value, 'shape': shape}
    try:
        op = graph._create_op_internal('Placeholder', [], [dtype], input_types=[], attrs=attrs, name=name)
    except ValueError as e:
        logging.warning(e)
        op = graph._create_op_internal('Placeholder', [], [dtype], input_types=[], attrs=attrs)
    result, = op.outputs
    if op_callbacks.should_invoke_op_callbacks():
        callback_outputs = op_callbacks.invoke_op_callbacks('Placeholder', tuple(), attrs, tuple(op.outputs), op_name=name, graph=graph)
        if callback_outputs is not None:
            result, = callback_outputs
    return result