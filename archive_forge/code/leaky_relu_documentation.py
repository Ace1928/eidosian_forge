from keras.src import backend
from keras.src.engine.base_layer import Layer
from keras.src.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export
Leaky version of a Rectified Linear Unit.

    It allows a small gradient when the unit is not active:

    ```
        f(x) = alpha * x if x < 0
        f(x) = x if x >= 0
    ```

    Usage:

    >>> layer = tf.keras.layers.LeakyReLU()
    >>> output = layer([-3.0, -1.0, 0.0, 2.0])
    >>> list(output.numpy())
    [-0.9, -0.3, 0.0, 2.0]
    >>> layer = tf.keras.layers.LeakyReLU(alpha=0.1)
    >>> output = layer([-3.0, -1.0, 0.0, 2.0])
    >>> list(output.numpy())
    [-0.3, -0.1, 0.0, 2.0]

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the batch axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as the input.

    Args:
        alpha: Float >= `0.`. Negative slope coefficient. Defaults to `0.3`.

    