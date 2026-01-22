from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sets
from tensorflow.python.util.tf_export import tf_export
def _has_valid_dims(weights_shape, values_shape):
    with ops.name_scope(None, 'has_invalid_dims', (weights_shape, values_shape)) as scope:
        values_shape_2d = array_ops.expand_dims(values_shape, -1)
        valid_dims = array_ops.concat((values_shape_2d, array_ops.ones_like(values_shape_2d)), axis=1)
        weights_shape_2d = array_ops.expand_dims(weights_shape, -1)
        invalid_dims = sets.set_difference(weights_shape_2d, valid_dims)
        num_invalid_dims = array_ops.size(invalid_dims.values, name='num_invalid_dims')
        return math_ops.equal(0, num_invalid_dims, name=scope)