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
class FrequencyEstimatorParameters(_OptimizationParameters):
    """Optimization parameters for Frequency Estimator TPU embeddings.

  This is a non-standard optimizer, which returns the estimated frequency of
  lookup for the feature passed to it. It should only be used on a table of
  width 1. The gradient fed back to the TPU embedding should always be zero.
  This can be acomplished via using `tf.stop_gradients` on the feature before
  using it.

  You must use the dynamic learning rate mechanism to set the 'learning rate'
  for this table to be the a float32 cast of the global training step counter.

  See `tensorflow/core/protobuf/tpu/optimization_parameters.proto` for more
  details on this optimizer.

  Pass this to `tf.estimator.tpu.experimental.EmbeddingConfigSpec` via the
  `optimization_parameters` argument to set the optimizer and its parameters.
  See the documentation for `tf.estimator.tpu.experimental.EmbeddingConfigSpec`
  for more details.

  ```
  estimator = tf.estimator.tpu.TPUEstimator(
      ...
      embedding_spec=tf.estimator.tpu.experimental.EmbeddingConfigSpec(
          ...
          optimization_parameters=FrequencyEstimatorParameters(0.1),
          ...))
  ```

  """

    def __init__(self, tau: float, max_delta: float, outlier_threshold: float, weight_exponent: float):
        """Optimization parameters for frequency estimator.

    Args:
      tau: Learning rate between (0, 1) that is used to update the array.
      max_delta: Maximum value of delta, the difference between the current
        global step and the last global step at which the row was sampled.
      outlier_threshold: Threshold used to determine whether the current update
        is an outlier.
      weight_exponent: The weight exponent used to transform the estimated delta
        into weights.
    """
        super().__init__(learning_rate=1.0, use_gradient_accumulation=True, clip_weight_min=None, clip_weight_max=None, weight_decay_factor=None, multiply_weight_decay_factor_by_learning_rate=None)
        self.tau = tau
        self.max_delta = max_delta
        self.outlier_threshold = outlier_threshold
        self.weight_exponent = weight_exponent