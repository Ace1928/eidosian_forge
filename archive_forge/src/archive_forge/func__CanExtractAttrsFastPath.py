from google.protobuf import text_format
from tensorflow.core.config import flags
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import op_def_library_pybind
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util import tf_contextlib
def _CanExtractAttrsFastPath(op_def, keywords):
    """Check if the fast path for _apply_op_helper is applicable."""
    for input_arg in op_def.input_arg:
        value = keywords.get(input_arg.name, None)
        if not isinstance(value, tensor.Tensor):
            return False
    for attr_def in op_def.attr:
        if attr_def.type == 'func' or attr_def.type == 'list(func)':
            return False
    return True