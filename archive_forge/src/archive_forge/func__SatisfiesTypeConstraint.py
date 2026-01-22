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
def _SatisfiesTypeConstraint(dtype, attr_def, param_name):
    if attr_def.HasField('allowed_values'):
        allowed_list = attr_def.allowed_values.list.type
        allowed_values = ', '.join((dtypes.as_dtype(x).name for x in allowed_list))
        if dtype not in allowed_list:
            raise TypeError(f"Value passed to parameter '{param_name}' has DataType {dtypes.as_dtype(dtype).name} not in list of allowed values: {allowed_values}")