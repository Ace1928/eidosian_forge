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
def _check_batch_shape_possibly_add_asserts(self):
    """Static check of init arg `batch_shape`, possibly add asserts."""
    if self._batch_shape_arg is None:
        return
    if self._assert_proper_shapes:
        self._batch_shape_arg = control_flow_ops.with_dependencies([check_ops.assert_rank(self._batch_shape_arg, 1, message='Argument batch_shape must be a 1-D Tensor.'), check_ops.assert_non_negative(self._batch_shape_arg, message='Argument batch_shape must be non-negative.')], self._batch_shape_arg)
    if not self._batch_shape_arg.dtype.is_integer:
        raise TypeError('Argument batch_shape must be integer type.  Found: %s' % self._batch_shape_arg)
    if self._batch_shape_static is None:
        return
    if self._batch_shape_static.ndim != 1:
        raise ValueError('Argument batch_shape must be a 1-D Tensor.  Found: %s' % self._batch_shape_static)
    if np.any(self._batch_shape_static < 0):
        raise ValueError('Argument batch_shape must be non-negative.  Found:%s' % self._batch_shape_static)