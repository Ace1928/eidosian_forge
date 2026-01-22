import contextlib
from tensorflow.compiler.jit.ops import xla_ops
from tensorflow.compiler.jit.ops import xla_ops_grad  # pylint: disable=unused-import
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.distribute import summary_op_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
def _RemoveExternalControlEdges(self, op):
    """Remove any external control dependency on this op."""
    internal_control_inputs = []
    external_control_inputs = []
    for x in op.control_inputs:
        is_internal_op = False
        ctxt = x._get_control_flow_context()
        while ctxt is not None:
            if ctxt == self:
                is_internal_op = True
                break
            ctxt = ctxt._outer_context
        if is_internal_op:
            internal_control_inputs.append(x)
        else:
            external_control_inputs.append(x)
    op._remove_all_control_inputs()
    op._add_control_inputs(internal_control_inputs)
    return (internal_control_inputs, external_control_inputs)