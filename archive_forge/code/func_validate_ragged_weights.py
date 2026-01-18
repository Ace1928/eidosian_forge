from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import bincount_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_count_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import dispatch
def validate_ragged_weights(values, weights, dtype=None):
    """Validates the passed weight tensor or creates an empty one."""
    if weights is None:
        if dtype:
            return array_ops.constant([], dtype=dtype)
        return array_ops.constant([], dtype=values.values.dtype)
    if not isinstance(weights, ragged_tensor.RaggedTensor):
        raise ValueError(f'`weights` must be a RaggedTensor if `values` is a RaggedTensor. Received argument weights={weights} of type: {type(weights).__name__}.')
    checks = []
    if weights.row_splits is not values.row_splits:
        checks.append(check_ops.assert_equal(weights.row_splits, values.row_splits, message="'weights' and 'values' must have the same row splits."))
    if checks:
        with ops.control_dependencies(checks):
            weights = array_ops.identity(weights.values)
    else:
        weights = weights.values
    return weights