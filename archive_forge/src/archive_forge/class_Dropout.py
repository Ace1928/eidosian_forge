import warnings
from tensorflow.python.keras import layers as keras_layers
from tensorflow.python.keras.legacy_tf_layers import base
from tensorflow.python.ops import init_ops
class Dropout(keras_layers.Dropout, base.Layer):
    """Applies Dropout to the input.

  Dropout consists in randomly setting a fraction `rate` of input units to 0
  at each update during training time, which helps prevent overfitting.
  The units that are kept are scaled by `1 / (1 - rate)`, so that their
  sum is unchanged at training time and inference time.

  Args:
    rate: The dropout rate, between 0 and 1. E.g. `rate=0.1` would drop out
      10% of input units.
    noise_shape: 1D tensor of type `int32` representing the shape of the
      binary dropout mask that will be multiplied with the input.
      For instance, if your inputs have shape
      `(batch_size, timesteps, features)`, and you want the dropout mask
      to be the same for all timesteps, you can use
      `noise_shape=[batch_size, 1, features]`.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed`.
      for behavior.
    name: The name of the layer (string).
  """

    def __init__(self, rate=0.5, noise_shape=None, seed=None, name=None, **kwargs):
        super(Dropout, self).__init__(rate=rate, noise_shape=noise_shape, seed=seed, name=name, **kwargs)

    def call(self, inputs, training=False):
        return super(Dropout, self).call(inputs, training=training)