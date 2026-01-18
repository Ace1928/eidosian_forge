import abc
import contextlib
import functools
import itertools
import math
import random
import numpy as np
from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import dataset_creator
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
def unpack_x_y_sample_weight(data):
    """Unpacks user-provided data tuple.

  This is a convenience utility to be used when overriding
  `Model.train_step`, `Model.test_step`, or `Model.predict_step`.
  This utility makes it easy to support data of the form `(x,)`,
  `(x, y)`, or `(x, y, sample_weight)`.

  Standalone usage:

  >>> features_batch = tf.ones((10, 5))
  >>> labels_batch = tf.zeros((10, 5))
  >>> data = (features_batch, labels_batch)
  >>> # `y` and `sample_weight` will default to `None` if not provided.
  >>> x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
  >>> sample_weight is None
  True

  Example in overridden `Model.train_step`:

  ```python
  class MyModel(tf.keras.Model):

    def train_step(self, data):
      # If `sample_weight` is not provided, all samples will be weighted
      # equally.
      x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)

      with tf.GradientTape() as tape:
        y_pred = self(x, training=True)
        loss = self.compiled_loss(
          y, y_pred, sample_weight, regularization_losses=self.losses)
        trainable_variables = self.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

      self.compiled_metrics.update_state(y, y_pred, sample_weight)
      return {m.name: m.result() for m in self.metrics}
  ```

  Args:
    data: A tuple of the form `(x,)`, `(x, y)`, or `(x, y, sample_weight)`.

  Returns:
    The unpacked tuple, with `None`s for `y` and `sample_weight` if they are not
    provided.
  """
    if not isinstance(data, tuple):
        return (data, None, None)
    elif len(data) == 1:
        return (data[0], None, None)
    elif len(data) == 2:
        return (data[0], data[1], None)
    elif len(data) == 3:
        return (data[0], data[1], data[2])
    else:
        error_msg = 'Data is expected to be in format `x`, `(x,)`, `(x, y)`, or `(x, y, sample_weight)`, found: {}'.format(data)
        raise ValueError(error_msg)