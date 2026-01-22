from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
def _calculate_scores(self, query, key):
    """Calculates attention scores as a nonlinear sum of query and key.

    Args:
      query: Query tensor of shape `[batch_size, Tq, dim]`.
      key: Key tensor of shape `[batch_size, Tv, dim]`.
    Returns:
      Tensor of shape `[batch_size, Tq, Tv]`.
    """
    q_reshaped = array_ops.expand_dims(query, axis=-2)
    k_reshaped = array_ops.expand_dims(key, axis=-3)
    if self.use_scale:
        scale = self.scale
    else:
        scale = 1.0
    return math_ops.reduce_sum(scale * math_ops.tanh(q_reshaped + k_reshaped), axis=-1)