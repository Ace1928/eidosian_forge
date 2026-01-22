import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
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
def _ones_diag(self):
    """Returns the diagonal of this operator as all ones."""
    if self.shape.is_fully_defined():
        d_shape = self.batch_shape.concatenate([self._min_matrix_dim()])
    else:
        d_shape = array_ops.concat([self.batch_shape_tensor(), [self._min_matrix_dim_tensor()]], axis=0)
    return array_ops.ones(shape=d_shape, dtype=self.dtype)