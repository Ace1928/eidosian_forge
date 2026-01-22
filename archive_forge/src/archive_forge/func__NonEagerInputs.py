import collections
import contextlib
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_state
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util import variable_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def _NonEagerInputs(op, xs_set):
    """Returns the inputs of op, crossing closure boundaries where necessary.

  Does not return any captured EagerTensors, i.e., the number of tensors
  returned may be less than the actual number of inputs.

  Args:
    op: Operation
    xs_set: ObjectIdentitySet of Tensors we are differentiating w.r.t.

  Returns:
    A list of tensors. The tensors may be from multiple Graph/FuncGraphs if op
    is in a FuncGraph and has captured inputs.
  """
    return [t for t in _Inputs(op, xs_set) if not isinstance(t, ops.EagerTensor)]