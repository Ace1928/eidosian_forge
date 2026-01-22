import numpy as np
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.profiler import trace
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util.tf_export import tf_export
def _eager_reshape(tensor, shape, ctx):
    """Eager-only version of Reshape op; requires tensor is an eager Tensor."""
    attr_t = tensor._datatype_enum()
    attr_tshape, (shape,) = execute.args_to_matching_eager([shape], ctx, [dtypes.int32, dtypes.int64], dtypes.int32)
    inputs_flat = [tensor, shape]
    attrs = ('T', attr_t, 'Tshape', attr_tshape)
    [result] = execute.execute(b'Reshape', 1, inputs=inputs_flat, attrs=attrs, ctx=ctx)
    return result