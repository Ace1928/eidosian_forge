import re
import uuid
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import variables
@ops.RegisterGradient('DebugGradientIdentity')
def _identify_gradient_grad(op, dy):
    """Gradient function for the DebugIdentity op."""
    grad_debugger_uuid, orig_tensor_name = _parse_grad_debug_op_name(op.name)
    grad_debugger = _gradient_debuggers[grad_debugger_uuid]
    grad_debugger.register_gradient_tensor(orig_tensor_name, dy)
    return dy