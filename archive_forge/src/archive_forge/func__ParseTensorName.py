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
def _ParseTensorName(tensor_name):
    """Parses a tensor name into an operation name and output index.

  This function will canonicalize tensor names as follows:

  * "foo:0"       -> ("foo", 0)
  * "foo:7"       -> ("foo", 7)
  * "foo"         -> ("foo", 0)
  * "foo:bar:baz" -> ValueError

  Args:
    tensor_name: The name of a tensor.

  Returns:
    A tuple containing the operation name, and the output index.

  Raises:
    ValueError: If `tensor_name' cannot be interpreted as the name of a tensor.
  """
    components = tensor_name.split(':')
    if len(components) == 2:
        try:
            output_index = int(components[1])
        except ValueError:
            raise ValueError(f'Cannot convert {tensor_name!r} to a tensor name. Second component of the name following the `:` should be an int. Got {components[1]}.')
        return (components[0], output_index)
    elif len(components) == 1:
        return (components[0], 0)
    else:
        raise ValueError(f"Cannot convert '{tensor_name}' to a tensor name. Tensor names should not contain more than 1 `:`. Obtained {len(components) - 1}")