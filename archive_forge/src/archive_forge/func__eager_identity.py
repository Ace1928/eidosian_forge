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
def _eager_identity(tensor, ctx):
    """Eager-only version of Identity op; requires tensor is an eager Tensor."""
    attrs = ('T', tensor.dtype.as_datatype_enum)
    [result] = execute.execute(b'Identity', 1, inputs=[tensor], attrs=attrs, ctx=ctx)
    return result