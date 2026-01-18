import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine.base_layer import Layer
from keras.src.engine.input_spec import InputSpec
from keras.src.utils import conv_utils
from tensorflow.python.util.tf_export import keras_export
Upsampling layer for 3D inputs.

    Repeats the 1st, 2nd and 3rd dimensions
    of the data by `size[0]`, `size[1]` and `size[2]` respectively.

    Examples:

    >>> input_shape = (2, 1, 2, 1, 3)
    >>> x = tf.constant(1, shape=input_shape)
    >>> y = tf.keras.layers.UpSampling3D(size=2)(x)
    >>> print(y.shape)
    (2, 2, 4, 2, 3)

    Args:
      size: Int, or tuple of 3 integers.
        The upsampling factors for dim1, dim2 and dim3.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
        while `channels_first` corresponds to inputs with shape
        `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
        When unspecified, uses
        `image_data_format` value found in your Keras config file at
         `~/.keras/keras.json` (if exists) else 'channels_last'.
        Defaults to 'channels_last'.

    Input shape:
      5D tensor with shape:
      - If `data_format` is `"channels_last"`:
          `(batch_size, dim1, dim2, dim3, channels)`
      - If `data_format` is `"channels_first"`:
          `(batch_size, channels, dim1, dim2, dim3)`

    Output shape:
      5D tensor with shape:
      - If `data_format` is `"channels_last"`:
          `(batch_size, upsampled_dim1, upsampled_dim2, upsampled_dim3,
          channels)`
      - If `data_format` is `"channels_first"`:
          `(batch_size, channels, upsampled_dim1, upsampled_dim2,
          upsampled_dim3)`
    