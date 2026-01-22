import contextlib
from tensorflow.core.framework import graph_pb2
from tensorflow.python import tf2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import control_flow_util
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export
def _ProcessInputMapParam(input_map):
    """Type-checks and possibly canonicalizes `input_map`."""
    if input_map is None:
        input_map = {}
    else:
        if not isinstance(input_map, dict):
            raise TypeError(f'Argument `input_map` must be a dictionary. Obtained {type(input_map).__name__}')
        if not all((isinstance(k, compat.bytes_or_text_types) for k in input_map.keys())):
            raise TypeError(f'All keys for argument `input_map` must be strings. Obtained keys: {list(input_map.keys())}')
    return input_map