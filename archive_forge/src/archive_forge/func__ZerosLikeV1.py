from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util as util
from tensorflow.python.ops import control_flow_v2_func_graphs
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import resource_variable_ops
def _ZerosLikeV1(op, index):
    """Branch of ZerosLike for TF1."""
    val = op.outputs[index]
    op_ctxt = op._get_control_flow_context()
    if op_ctxt:
        pred = op_ctxt.pred
        branch = op_ctxt.branch
        switch_val = control_flow_ops.switch(op.inputs[0], pred)[1 - branch]
        pivot = array_ops.identity(switch_val)
        if val.dtype == dtypes.resource:
            with ops.control_dependencies([pivot]):
                return array_ops.zeros(gen_resource_variable_ops.variable_shape(switch_val), dtype=default_gradient.get_zeros_dtype(val))
        zeros_shape = array_ops.shape_internal(switch_val, optimize=False)
        with ops.control_dependencies([pivot]):
            return array_ops.zeros(zeros_shape, dtype=val.dtype)
    else:
        return array_ops.zeros_like(val, optimize=False)