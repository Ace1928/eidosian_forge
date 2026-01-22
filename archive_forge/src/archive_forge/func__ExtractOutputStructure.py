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
def _ExtractOutputStructure(op_type_name, op_def, attr_protos, output_structure):
    """Extracts `output_structure`. For use in _apply_op_helper."""
    for arg in op_def.output_arg:
        if arg.number_attr:
            n = _AttrValue(attr_protos, arg.number_attr, op_type_name).i
            output_structure.append(n)
        elif arg.type_attr:
            t = _AttrValue(attr_protos, arg.type_attr, op_type_name)
            output_structure.append(None)
        elif arg.type_list_attr:
            t = _AttrValue(attr_protos, arg.type_list_attr, op_type_name)
            output_structure.append(len(t.list.type))
        else:
            output_structure.append(None)