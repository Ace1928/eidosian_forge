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
def _MakeFunc(v, arg_name):
    """Ensure v is a func."""
    if isinstance(v, attr_value_pb2.NameAttrList):
        return v
    if isinstance(v, compat.bytes_or_text_types):
        fn_attr = attr_value_pb2.NameAttrList(name=v)
    elif hasattr(v, 'add_to_graph'):
        v.add_to_graph(ops.get_default_graph())
        if hasattr(v, '_as_name_attr_list'):
            fn_attr = v._as_name_attr_list
        else:
            fn_attr = attr_value_pb2.NameAttrList(name=v.name)
    else:
        raise TypeError(f"Don't know how to convert {repr(v)} to a func for argument {arg_name}")
    return fn_attr