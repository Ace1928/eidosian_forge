from keras.src import backend
from keras.src.engine.base_layer import Layer
from keras.src.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export
Rectified Linear Unit activation function.

    With default values, it returns element-wise `max(x, 0)`.

    Otherwise, it follows:

    ```
        f(x) = max_value if x >= max_value
        f(x) = x if threshold <= x < max_value
        f(x) = negative_slope * (x - threshold) otherwise
    ```

    Usage:

    >>> layer = tf.keras.layers.ReLU()
    >>> output = layer([-3.0, -1.0, 0.0, 2.0])
    >>> list(output.numpy())
    [0.0, 0.0, 0.0, 2.0]
    >>> layer = tf.keras.layers.ReLU(max_value=1.0)
    >>> output = layer([-3.0, -1.0, 0.0, 2.0])
    >>> list(output.numpy())
    [0.0, 0.0, 0.0, 1.0]
    >>> layer = tf.keras.layers.ReLU(negative_slope=1.0)
    >>> output = layer([-3.0, -1.0, 0.0, 2.0])
    >>> list(output.numpy())
    [-3.0, -1.0, 0.0, 2.0]
    >>> layer = tf.keras.layers.ReLU(threshold=1.5)
    >>> output = layer([-3.0, -1.0, 1.0, 2.0])
    >>> list(output.numpy())
    [0.0, 0.0, 0.0, 2.0]

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the batch axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as the input.

    Args:
        max_value: Float >= 0. Maximum activation value. None means unlimited.
            Defaults to `None`.
        negative_slope: Float >= 0. Negative slope coefficient.
            Defaults to `0.`.
        threshold: Float >= 0. Threshold value for thresholded activation.
            Defaults to `0.`.
    