from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sets
from tensorflow.python.util.tf_export import tf_export
def _has_valid_nonscalar_shape(weights_rank, weights_shape, values_rank, values_shape):
    with ops.name_scope(None, 'has_valid_nonscalar_shape', (weights_rank, weights_shape, values_rank, values_shape)) as scope:
        is_same_rank = math_ops.equal(values_rank, weights_rank, name='is_same_rank')
        return cond.cond(is_same_rank, lambda: _has_valid_dims(weights_shape, values_shape), lambda: is_same_rank, name=scope)