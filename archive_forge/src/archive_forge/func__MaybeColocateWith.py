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
@tf_contextlib.contextmanager
def _MaybeColocateWith(inputs):
    """A context manager for (maybe) colocating with a list of input tensors.

  Args:
    inputs: A list of `Tensor` or `Operation` objects.

  Returns:
    A context manager.
  """
    if not inputs:
        yield
    else:
        with ops.colocate_with(inputs[0]), _MaybeColocateWith(inputs[1:]):
            yield