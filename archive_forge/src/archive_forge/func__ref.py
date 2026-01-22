from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import variable_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated
def _ref(self):
    """Returns a reference to this variable.

    You usually do not need to call this method as all ops that need a reference
    to the variable call it automatically.

    Returns is a `Tensor` which holds a reference to the variable.  You can
    assign a new value to the variable by passing the tensor to an assign op.
    See `tf.Variable.value` if you want to get the value of the
    variable.

    Returns:
      A `Tensor` that is a reference to the variable.
    """
    return self._variable