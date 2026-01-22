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
class CosineSimilarity(LossFunctionWrapper):
    """Computes the cosine similarity between labels and predictions.

  Note that it is a number between -1 and 1. When it is a negative number
  between -1 and 0, 0 indicates orthogonality and values closer to -1
  indicate greater similarity. The values closer to 1 indicate greater
  dissimilarity. This makes it usable as a loss function in a setting
  where you try to maximize the proximity between predictions and targets.
  If either `y_true` or `y_pred` is a zero vector, cosine similarity will be 0
  regardless of the proximity between predictions and targets.

  `loss = -sum(l2_norm(y_true) * l2_norm(y_pred))`

  Standalone usage:

  >>> y_true = [[0., 1.], [1., 1.]]
  >>> y_pred = [[1., 0.], [1., 1.]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
  >>> # l2_norm(y_true) = [[0., 1.], [1./1.414, 1./1.414]]
  >>> # l2_norm(y_pred) = [[1., 0.], [1./1.414, 1./1.414]]
  >>> # l2_norm(y_true) . l2_norm(y_pred) = [[0., 0.], [0.5, 0.5]]
  >>> # loss = mean(sum(l2_norm(y_true) . l2_norm(y_pred), axis=1))
  >>> #       = -((0. + 0.) +  (0.5 + 0.5)) / 2
  >>> cosine_loss(y_true, y_pred).numpy()
  -0.5

  >>> # Calling with 'sample_weight'.
  >>> cosine_loss(y_true, y_pred, sample_weight=[0.8, 0.2]).numpy()
  -0.0999

  >>> # Using 'sum' reduction type.
  >>> cosine_loss = tf.keras.losses.CosineSimilarity(axis=1,
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> cosine_loss(y_true, y_pred).numpy()
  -0.999

  >>> # Using 'none' reduction type.
  >>> cosine_loss = tf.keras.losses.CosineSimilarity(axis=1,
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> cosine_loss(y_true, y_pred).numpy()
  array([-0., -0.999], dtype=float32)

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tf.keras.losses.CosineSimilarity(axis=1))
  ```

  Args:
    axis: The axis along which the cosine similarity is computed
      (the features axis). Defaults to -1.
    reduction: Type of `tf.keras.losses.Reduction` to apply to loss.
      Default value is `AUTO`. `AUTO` indicates that the reduction option will
      be determined by the usage context. For almost all cases this defaults to
      `SUM_OVER_BATCH_SIZE`. When used with `tf.distribute.Strategy`, outside of
      built-in training loops such as `tf.keras` `compile` and `fit`, using
      `AUTO` or `SUM_OVER_BATCH_SIZE` will raise an error. Please see this
      custom training [tutorial]
      (https://www.tensorflow.org/tutorials/distribute/custom_training) for more
        details.
    name: Optional name for the instance.
  """

    def __init__(self, axis=-1, reduction=losses_utils.ReductionV2.AUTO, name='cosine_similarity'):
        super().__init__(cosine_similarity, reduction=reduction, name=name, axis=axis)