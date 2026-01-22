import math
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops
class GlorotUniform(VarianceScaling):
    """The Glorot uniform initializer, also called Xavier uniform initializer.

  Also available via the shortcut function
  `tf.keras.initializers.glorot_uniform`.

  Draws samples from a uniform distribution within `[-limit, limit]`, where
  `limit = sqrt(6 / (fan_in + fan_out))` (`fan_in` is the number of input units
  in the weight tensor and `fan_out` is the number of output units).

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.GlorotUniform()
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.GlorotUniform()
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

  Args:
    seed: A Python integer. An initializer created with a given seed will
      always produce the same random tensor for a given shape and dtype.

  References:
      [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)
      ([pdf](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf))
  """

    def __init__(self, seed=None):
        super(GlorotUniform, self).__init__(scale=1.0, mode='fan_avg', distribution='uniform', seed=seed)

    def get_config(self):
        return {'seed': self.seed}