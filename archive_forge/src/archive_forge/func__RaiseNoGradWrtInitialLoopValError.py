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
def _RaiseNoGradWrtInitialLoopValError(op, from_ops, xs_set):
    """Raises an error if we backprop through a loop var."""
    target_op = None
    queue = collections.deque([op])
    visited = set()
    while queue:
        curr_op = queue.popleft()
        if curr_op in visited:
            continue
        visited.add(curr_op)
        if curr_op in from_ops:
            target_op = curr_op
            break
        queue.extend((t.op for t in _NonEagerInputs(curr_op, xs_set)))
    assert target_op
    raise ValueError(f"Cannot compute gradient inside while loop with respect to op '{target_op.name}'. We do not support taking the gradient wrt or through the initial value of a loop variable. Gradients can be computed through loop invariants or wrt the input parameters to the loop body.")