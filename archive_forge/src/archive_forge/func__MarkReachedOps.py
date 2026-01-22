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
def _MarkReachedOps(from_ops, reached_ops, func_graphs):
    """Mark all ops reached from "from_ops".

  Args:
    from_ops: list of Operations.
    reached_ops: set of Operations.
    func_graphs: list of FuncGraphs. This method will traverse through
      these functions if they capture from_ops or any reachable ops.
  """
    queue = collections.deque()
    queue.extend(from_ops)
    while queue:
        op = queue.popleft()
        if op not in reached_ops:
            reached_ops.add(op)
            for output in op.outputs:
                if backprop_util.IsTrainable(output):
                    queue.extend(_Consumers(output, func_graphs))