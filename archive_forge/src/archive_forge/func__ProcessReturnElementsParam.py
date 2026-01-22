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
def _ProcessReturnElementsParam(return_elements):
    """Type-checks and possibly canonicalizes `return_elements`."""
    if return_elements is None:
        return None
    if not all((isinstance(x, compat.bytes_or_text_types) for x in return_elements)):
        raise TypeError(f'Argument `return_elements` must be a list of strings. Obtained {return_elements}.')
    return tuple((compat.as_str(x) for x in return_elements))