import collections
import copy
import math
import re
from typing import Optional
from tensorflow.core.protobuf.tpu import optimization_parameters_pb2
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2 as elc
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['tpu.experimental.FtrlParameters'])
class FtrlParameters(_OptimizationParameters):
    """Optimization parameters for Ftrl with TPU embeddings.

  Pass this to `tf.estimator.tpu.experimental.EmbeddingConfigSpec` via the
  `optimization_parameters` argument to set the optimizer and its parameters.
  See the documentation for `tf.estimator.tpu.experimental.EmbeddingConfigSpec`
  for more details.

  ```
  estimator = tf.estimator.tpu.TPUEstimator(
      ...
      embedding_config_spec=tf.estimator.tpu.experimental.EmbeddingConfigSpec(
          ...
          optimization_parameters=tf.tpu.experimental.FtrlParameters(0.1),
          ...))
  ```

  """

    def __init__(self, learning_rate: float, learning_rate_power: float=-0.5, initial_accumulator_value: float=0.1, l1_regularization_strength: float=0.0, l2_regularization_strength: float=0.0, use_gradient_accumulation: bool=True, clip_weight_min: Optional[float]=None, clip_weight_max: Optional[float]=None, weight_decay_factor: Optional[float]=None, multiply_weight_decay_factor_by_learning_rate: Optional[bool]=None, multiply_linear_by_learning_rate: bool=False, beta: float=0, allow_zero_accumulator: bool=False, clip_gradient_min: Optional[float]=None, clip_gradient_max: Optional[float]=None):
        """Optimization parameters for Ftrl.

    Implements FTRL as described in the following [paper](
    https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf)

    Args:
      learning_rate: a floating point value. The learning rate.
      learning_rate_power: A float value, must be less or equal to zero.
        Controls how the learning rate decreases during training. Use zero for a
        fixed learning rate. See section 3.1 in the
        [paper](https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf).
      initial_accumulator_value: The starting value for accumulators. Only zero
        or positive values are allowed.
      l1_regularization_strength: A float value, must be greater than or equal
        to zero.
      l2_regularization_strength: A float value, must be greater than or equal
        to zero.
      use_gradient_accumulation: setting this to `False` makes embedding
        gradients calculation less accurate but faster. Please see
        `optimization_parameters.proto` for details. for details.
      clip_weight_min: the minimum value to clip by; None means -infinity.
      clip_weight_max: the maximum value to clip by; None means +infinity.
      weight_decay_factor: amount of weight decay to apply; None means that the
        weights are not decayed.
      multiply_weight_decay_factor_by_learning_rate: if true,
        `weight_decay_factor` is multiplied by the current learning rate.
      multiply_linear_by_learning_rate: When true, multiplies the usages of the
        linear slot in the weight update by the learning rate. This is useful
        when ramping up learning rate from 0 (which would normally produce
        NaNs).
      beta: The beta parameter for FTRL.
      allow_zero_accumulator: Changes the implementation of the square root to
        allow for the case of initial_accumulator_value being zero. This will
        cause a slight performance drop.
      clip_gradient_min: the minimum value to clip by; None means -infinity.
        Gradient accumulation must be set to true if this is set.
      clip_gradient_max: the maximum value to clip by; None means +infinity.
        Gradient accumulation must be set to true if this is set.
    """
        super().__init__(learning_rate=learning_rate, use_gradient_accumulation=use_gradient_accumulation, clip_weight_min=clip_weight_min, clip_weight_max=clip_weight_max, weight_decay_factor=weight_decay_factor, multiply_weight_decay_factor_by_learning_rate=multiply_weight_decay_factor_by_learning_rate, clip_gradient_min=clip_gradient_min, clip_gradient_max=clip_gradient_max)
        if learning_rate_power > 0.0:
            raise ValueError('learning_rate_power must be less than or equal to 0. got {}.'.format(learning_rate_power))
        if initial_accumulator_value < 0.0:
            raise ValueError('initial_accumulator_value must be greater than or equal to 0. got {}.'.format(initial_accumulator_value))
        if l1_regularization_strength < 0.0:
            raise ValueError('l1_regularization_strength must be greater than or equal to 0. got {}.'.format(l1_regularization_strength))
        if l2_regularization_strength < 0.0:
            raise ValueError('l2_regularization_strength must be greater than or equal to 0. got {}.'.format(l2_regularization_strength))
        self.learning_rate_power = learning_rate_power
        self.initial_accumulator_value = initial_accumulator_value
        self.initial_linear_value = 0.0
        self.l1_regularization_strength = l1_regularization_strength
        self.l2_regularization_strength = l2_regularization_strength
        self.multiply_linear_by_learning_rate = multiply_linear_by_learning_rate
        self.beta = beta
        self.allow_zero_accumulator = allow_zero_accumulator