from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.util.tf_export import tf_export
def _solve_matmul_internal(self, x, solve_matmul_fn, adjoint=False, adjoint_arg=False):
    output = x
    if adjoint_arg:
        if self.dtype.is_complex:
            output = math_ops.conj(output)
    else:
        output = linalg.transpose(output)
    for o in reversed(self.operators):
        if adjoint:
            operator_dimension = o.range_dimension_tensor()
        else:
            operator_dimension = o.domain_dimension_tensor()
        output_shape = _prefer_static_shape(output)
        if tensor_util.constant_value(operator_dimension) is not None:
            operator_dimension = tensor_util.constant_value(operator_dimension)
            if output.shape[-2] is not None and output.shape[-1] is not None:
                dim = int(output.shape[-2] * output_shape[-1] // operator_dimension)
        else:
            dim = math_ops.cast(output_shape[-2] * output_shape[-1] // operator_dimension, dtype=dtypes.int32)
        output_shape = _prefer_static_concat_shape(output_shape[:-2], [dim, operator_dimension])
        output = array_ops.reshape(output, shape=output_shape)
        if self.dtype.is_complex:
            output = math_ops.conj(output)
        output = solve_matmul_fn(o, output, adjoint=adjoint, adjoint_arg=True)
    if adjoint_arg:
        col_dim = _prefer_static_shape(x)[-2]
    else:
        col_dim = _prefer_static_shape(x)[-1]
    if adjoint:
        row_dim = self.domain_dimension_tensor()
    else:
        row_dim = self.range_dimension_tensor()
    matrix_shape = [row_dim, col_dim]
    output = array_ops.reshape(output, _prefer_static_concat_shape(_prefer_static_shape(output)[:-2], matrix_shape))
    if x.shape.is_fully_defined():
        if adjoint_arg:
            column_dim = x.shape[-2]
        else:
            column_dim = x.shape[-1]
        broadcast_batch_shape = common_shapes.broadcast_shape(x.shape[:-2], self.batch_shape)
        if adjoint:
            matrix_dimensions = [self.domain_dimension, column_dim]
        else:
            matrix_dimensions = [self.range_dimension, column_dim]
        output.set_shape(broadcast_batch_shape.concatenate(matrix_dimensions))
    return output