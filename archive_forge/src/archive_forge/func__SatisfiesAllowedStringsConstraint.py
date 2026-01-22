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
def _SatisfiesAllowedStringsConstraint(value, attr_def, arg_name, op_type_name):
    if value not in attr_def.allowed_values.list.s:
        allowed_values = '", "'.join(map(compat.as_text, attr_def.allowed_values.list.s))
        raise ValueError(f'''Attr '{arg_name}' of '{op_type_name}' Op passed string '{compat.as_text(value)}' not in: "{allowed_values}".''')