from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.control_flow_ops import *
def _SwitchGrad(op, *grad):
    """Gradients for a Switch op is calculated using a Merge op.

  If the switch is a loop switch, it will be visited twice. We create
  the merge on the first visit, and update the other input of the merge
  on the second visit. A next_iteration is also added on second visit.
  """
    graph = ops.get_default_graph()
    op_ctxt = op._get_control_flow_context()
    grad_ctxt = graph._get_control_flow_context()
    if isinstance(op_ctxt, WhileContext):
        merge_grad = grad_ctxt.grad_state.switch_map.get(op)
        if merge_grad is not None:
            if grad[1] is not None:
                control_flow_ops._AddNextAndBackEdge(merge_grad, grad[1], enforce_shape_invariant=False)
            return (None, None)
        elif grad[0] is not None:
            merge_grad = merge([grad[0], grad[0]], name='b_switch')[0]
            grad_ctxt.grad_state.switch_map[op] = merge_grad
            return (merge_grad, None)
        else:
            return (None, None)
    elif isinstance(op_ctxt, CondContext):
        zero_grad = grad[1 - op_ctxt.branch]
        if zero_grad is None:
            if op.inputs[0].dtype == dtypes.resource:
                return (merge([grad[op_ctxt.branch]] * 2, name='cond_resource_grad')[0], None)
            return (None, None)
        return (merge(grad, name='cond_grad')[0], None)
    else:
        false_grad = switch(grad[0], op.inputs[1])[0]
        true_grad = switch(grad[1], op.inputs[1])[1]
        return (merge([false_grad, true_grad])[0], None)