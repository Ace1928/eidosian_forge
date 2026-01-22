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
@ops.RegisterGradient('MatrixSetDiagV3')
def _MatrixSetDiagGradV3(op, grad):
    """Gradient for MatrixSetDiagV3."""
    diag_shape = op.inputs[1].get_shape()
    align = op.get_attr('align')
    if not diag_shape.is_fully_defined():
        grad_shape = array_ops.shape(grad)
        batch_shape = grad_shape[:-2]
        matrix_shape = grad_shape[-2:]
        diag_index = array_ops.reshape(op.inputs[2], [-1])
        d_lower = diag_index[0]
        d_upper = diag_index[-1]
        y_offset = cond.cond(math_ops.less(d_upper, 0), lambda: d_upper, lambda: 0)
        x_offset = cond.cond(math_ops.greater(d_lower, 0), lambda: -d_lower, lambda: 0)
        max_diag_len = math_ops.minimum(matrix_shape[0] + y_offset, matrix_shape[1] + x_offset)
        postfix = cond.cond(math_ops.equal(d_lower, d_upper), lambda: ops.convert_to_tensor([max_diag_len]), lambda: ops.convert_to_tensor([d_upper - d_lower + 1, max_diag_len]))
        diag_shape = array_ops.concat([batch_shape, postfix], 0)
    grad_input = array_ops.matrix_set_diag(grad, array_ops.zeros(diag_shape, dtype=grad.dtype), k=op.inputs[2], align=align)
    grad_diag = array_ops.matrix_diag_part(grad, k=op.inputs[2], align=align)
    return (grad_input, grad_diag, None)