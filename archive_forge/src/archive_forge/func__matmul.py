import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util.tf_export import tf_export
def _matmul(self, x, adjoint=False, adjoint_arg=False):
    if self._assert_proper_shapes:
        x = linalg.adjoint(x) if adjoint_arg else x
        aps = linear_operator_util.assert_compatible_matrix_dimensions(self, x)
        x = control_flow_ops.with_dependencies([aps], x)
    if self.is_square:
        if adjoint_arg:
            output_shape = array_ops.concat([array_ops.shape(x)[:-2], [array_ops.shape(x)[-1], array_ops.shape(x)[-2]]], axis=0)
        else:
            output_shape = array_ops.shape(x)
        return self._possibly_broadcast_batch_shape(array_ops.zeros(shape=output_shape, dtype=x.dtype))
    x_shape = array_ops.shape(x)
    n = self._num_columns if adjoint else self._num_rows
    m = x_shape[-2] if adjoint_arg else x_shape[-1]
    output_shape = array_ops.concat([x_shape[:-2], [n, m]], axis=0)
    zeros = array_ops.zeros(shape=output_shape, dtype=x.dtype)
    return self._possibly_broadcast_batch_shape(zeros)