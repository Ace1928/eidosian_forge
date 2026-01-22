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
class Poisson(LossFunctionWrapper):
    """Computes the Poisson loss between `y_true` and `y_pred`.

  `loss = y_pred - y_true * log(y_pred)`

  Standalone usage:

  >>> y_true = [[0., 1.], [0., 0.]]
  >>> y_pred = [[1., 1.], [0., 0.]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> p = tf.keras.losses.Poisson()
  >>> p(y_true, y_pred).numpy()
  0.5

  >>> # Calling with 'sample_weight'.
  >>> p(y_true, y_pred, sample_weight=[0.8, 0.2]).numpy()
  0.4

  >>> # Using 'sum' reduction type.
  >>> p = tf.keras.losses.Poisson(
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> p(y_true, y_pred).numpy()
  0.999

  >>> # Using 'none' reduction type.
  >>> p = tf.keras.losses.Poisson(
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> p(y_true, y_pred).numpy()
  array([0.999, 0.], dtype=float32)

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tf.keras.losses.Poisson())
  ```
  """

    def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name='poisson'):
        """Initializes `Poisson` instance.

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
      name: Optional name for the instance. Defaults to 'poisson'.
    """
        super().__init__(poisson, name=name, reduction=reduction)