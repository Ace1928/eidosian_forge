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
def _MakeShape(v, arg_name):
    """Convert v into a TensorShapeProto."""
    if isinstance(v, tensor_shape_pb2.TensorShapeProto):
        for d in v.dim:
            if d.name:
                logging.warning('Warning: TensorShapeProto with a named dimension: %s', str(v))
                break
        return v
    try:
        return tensor_shape.as_shape(v).as_proto()
    except TypeError as e:
        raise TypeError(f'Error converting {repr(v)} (arg name = {arg_name}) to a TensorShape: {e}')
    except ValueError as e:
        raise TypeError(f'Error converting {repr(v)} (arg name = {arg_name}) to a TensorShape: {e}')