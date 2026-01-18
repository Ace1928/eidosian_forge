from keras.src import backend
from keras.src.layers.pooling.base_global_pooling2d import GlobalPooling2D
from tensorflow.python.util.tf_export import keras_export
Global max pooling operation for spatial data.

    Examples:

    >>> input_shape = (2, 4, 5, 3)
    >>> x = tf.random.normal(input_shape)
    >>> y = tf.keras.layers.GlobalMaxPooling2D()(x)
    >>> print(y.shape)
    (2, 3)

    Args:
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape
        `(batch, channels, height, width)`.
        When unspecified, uses
        `image_data_format` value found in your Keras config file at
         `~/.keras/keras.json` (if exists) else 'channels_last'.
        Defaults to 'channels_last'.
      keepdims: A boolean, whether to keep the spatial dimensions or not.
        If `keepdims` is `False` (default), the rank of the tensor is reduced
        for spatial dimensions.
        If `keepdims` is `True`, the spatial dimensions are retained with
        length 1.
        The behavior is the same as for `tf.reduce_max` or `np.max`.

    Input shape:
      - If `data_format='channels_last'`:
        4D tensor with shape `(batch_size, rows, cols, channels)`.
      - If `data_format='channels_first'`:
        4D tensor with shape `(batch_size, channels, rows, cols)`.

    Output shape:
      - If `keepdims`=False:
        2D tensor with shape `(batch_size, channels)`.
      - If `keepdims`=True:
        - If `data_format='channels_last'`:
          4D tensor with shape `(batch_size, 1, 1, channels)`
        - If `data_format='channels_first'`:
          4D tensor with shape `(batch_size, channels, 1, 1)`
    