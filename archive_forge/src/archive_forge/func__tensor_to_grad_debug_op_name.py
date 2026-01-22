import re
import uuid
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import variables
def _tensor_to_grad_debug_op_name(tensor, grad_debugger_uuid):
    op_name, slot = debug_graphs.parse_node_or_tensor_name(tensor.name)
    return '%s_%d/%s%s' % (op_name, slot, _GRADIENT_DEBUG_TAG, grad_debugger_uuid)