from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices as indexed_slices_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
@ops.RegisterGradient('StridedSlice')
def _StridedSliceGrad(op, grad):
    """Gradient for StridedSlice op."""
    begin = op.inputs[1]
    end = op.inputs[2]
    strides = op.inputs[3]
    x = array_ops.shape(op.inputs[0], out_type=begin.dtype)
    x_static = tensor_util.constant_value(x)
    x = x_static if x_static is not None else x
    begin_static = tensor_util.constant_value(begin)
    begin = begin_static if begin_static is not None else begin
    end_static = tensor_util.constant_value(end)
    end = end_static if end_static is not None else end
    strides_static = tensor_util.constant_value(strides)
    strides = strides_static if strides_static is not None else strides
    return (array_ops.strided_slice_grad(x, begin, end, strides, grad, begin_mask=op.get_attr('begin_mask'), end_mask=op.get_attr('end_mask'), ellipsis_mask=op.get_attr('ellipsis_mask'), new_axis_mask=op.get_attr('new_axis_mask'), shrink_axis_mask=op.get_attr('shrink_axis_mask')), None, None, None)