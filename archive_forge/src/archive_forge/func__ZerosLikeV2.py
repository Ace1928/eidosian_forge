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
def _ZerosLikeV2(op, index):
    """Branch of ZerosLike for TF2."""
    val = op.outputs[index]
    if val.dtype == dtypes.resource:
        return array_ops.zeros(gen_resource_variable_ops.variable_shape(val), dtype=default_gradient.get_zeros_dtype(val))
    if isinstance(val.op.graph, control_flow_v2_func_graphs.WhileBodyFuncGraph) and val.dtype != dtypes.variant:
        if val.shape.is_fully_defined():
            return constant_op.constant(0, shape=val.shape.dims, dtype=val.dtype)
        else:
            zeros_shape = array_ops.shape_internal(val, optimize=False)
            return array_ops.zeros(zeros_shape, val.dtype)
    else:
        return array_ops.zeros_like(val, optimize=False)