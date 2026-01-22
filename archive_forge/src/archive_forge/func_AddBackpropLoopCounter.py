import abc
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf import control_flow_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util as util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.gen_control_flow_ops import *
from tensorflow.python.util import compat
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def AddBackpropLoopCounter(self, count, outer_grad_state):
    """Add the backprop loop that controls the iterations.

    This is added to the backprop loop. It is used to control the loop
    termination of the backprop loop. Called in the outer context of
    this grad context.

    The pseudocode is:
      `n = count; while (n >= 1) { n--; }`

    Note that a control dependency is added to `final_zero` to ensure the
    correct execution order of stack pop ops.

    Args:
      count: The number of iterations for backprop.
      outer_grad_state: The outer grad state. None if not nested.

    Returns:
      The loop index.
    """
    in_separate_functions = count.graph is not ops.get_default_graph()
    if in_separate_functions:
        count = array_ops.identity(count)
    else:
        one = constant_op.constant(1, name='b_count')
    self.Enter()
    self.AddName(count.name)
    enter_count = _Enter(count, self._name, is_constant=False, parallel_iterations=self._parallel_iterations, name='b_count')
    self.loop_enters.append(enter_count)
    merge_count = merge([enter_count, enter_count])[0]
    self._pivot_for_pred = merge_count
    if in_separate_functions:
        one = constant_op.constant(1, name='b_count')
    pred = math_ops.greater_equal(merge_count, one)
    self._pivot = loop_cond(pred, name='b_count')
    switch_count = switch(merge_count, self._pivot)
    index = math_ops.subtract(switch_count[1], one)
    self._pivot_for_body = index
    next_count = _NextIteration(index)
    merge_count.op._update_input(1, next_count)
    final_zero = exit(switch_count[0], name='b_count')
    self.loop_exits.append(final_zero)
    if outer_grad_state is not None:
        outer_grad_state.grad_sync._add_control_input(final_zero.op)
    self.ExitResult([final_zero])
    self.Exit()
    return next_count