import abc
import functools
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.ops.ragged import ragged_map_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.util import dispatch
from tensorflow.tools.docs import doc_controls
class MeanSquaredLogarithmicError(LossFunctionWrapper):
    """Computes the mean squared logarithmic error between `y_true` and `y_pred`.

  `loss = square(log(y_true + 1.) - log(y_pred + 1.))`

  Standalone usage:

  >>> y_true = [[0., 1.], [0., 0.]]
  >>> y_pred = [[1., 1.], [1., 0.]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> msle = tf.keras.losses.MeanSquaredLogarithmicError()
  >>> msle(y_true, y_pred).numpy()
  0.240

  >>> # Calling with 'sample_weight'.
  >>> msle(y_true, y_pred, sample_weight=[0.7, 0.3]).numpy()
  0.120

  >>> # Using 'sum' reduction type.
  >>> msle = tf.keras.losses.MeanSquaredLogarithmicError(
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> msle(y_true, y_pred).numpy()
  0.480

  >>> # Using 'none' reduction type.
  >>> msle = tf.keras.losses.MeanSquaredLogarithmicError(
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> msle(y_true, y_pred).numpy()
  array([0.240, 0.240], dtype=float32)

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd',
                loss=tf.keras.losses.MeanSquaredLogarithmicError())
  ```
  """

    def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name='mean_squared_logarithmic_error'):
        """Initializes `MeanSquaredLogarithmicError` instance.

    Args:
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the instance. Defaults to
        'mean_squared_logarithmic_error'.
    """
        super().__init__(mean_squared_logarithmic_error, name=name, reduction=reduction)