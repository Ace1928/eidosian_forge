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
def _StopOps(from_ops, stop_gradient_ops, pending_count, xs_set):
    """The set of ops that terminate the gradient computation.

  This computes the frontier of the forward graph *before* which backprop
  should stop. Operations in the returned set will not be differentiated.
  This set is defined as the subset of `from_ops` containing ops that have
  no predecessor in `from_ops`. `pending_count` is the result of
  `_PendingCount(xs, from_ops)`. An 'op' has predecessors in `from_ops`
  iff pending_count[op] > 0.

  In addition, none of `stop_gradient_ops` will be differentiated.

  Args:
    from_ops: list of Operations.
    stop_gradient_ops: list of Operations never to backprop through.
    pending_count: mapping from operation to number of backprop inputs.
    xs_set: ObjectIdentitySet of Tensors.

  Returns:
    The set of operations.
  """
    stop_ops = set()
    for op in from_ops:
        is_stop_op = True
        for inp in _NonEagerInputs(op, xs_set):
            if pending_count[inp.op] > 0:
                is_stop_op = False
                break
        if is_stop_op:
            stop_ops.add(op)
    stop_ops.update((op for op in stop_gradient_ops))
    return stop_ops