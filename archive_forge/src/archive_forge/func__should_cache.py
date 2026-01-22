from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import while_loop
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _should_cache():
    """Returns True if a default caching device should be set, otherwise False."""
    if context.executing_eagerly():
        return False
    graph = ops.get_default_graph()
    ctxt = graph._get_control_flow_context()
    in_v1_while_loop = control_flow_util.GetContainingWhileContext(ctxt) is not None
    in_v2_while_loop = control_flow_util_v2.in_while_loop_defun(graph)
    return not in_v1_while_loop and (not in_v2_while_loop)