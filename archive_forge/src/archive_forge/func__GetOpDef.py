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
def _GetOpDef(op_type_name, keywords):
    """Returns the OpDef, Graph and Producer. For use in _apply_op_helper."""
    op_def = op_def_registry.get(op_type_name)
    if op_def is None:
        raise RuntimeError(f'Unrecognized Op name {op_type_name}')
    try:
        g = ops._get_graph_from_inputs(_Flatten(keywords.values()))
        producer = g.graph_def_versions.producer
    except AssertionError as e:
        raise RuntimeError(f"Cannot determine graph for Op '{op_type_name}' due to: {e.message}")
    return (op_def, g, producer)