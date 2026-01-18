from keras.src import activations
from keras.src import backend
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.engine.base_layer import Layer
from keras.src.engine.input_spec import InputSpec
from keras.src.layers.locally_connected import locally_connected_utils
from keras.src.utils import conv_utils
from keras.src.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export
Locally-connected layer for 2D inputs.

    The `LocallyConnected2D` layer works similarly
    to the `Conv2D` layer, except that weights are unshared,
    that is, a different set of filters is applied at each
    different patch of the input.

    Note: layer attributes cannot be modified after the layer has been called
    once (except the `trainable` attribute).

    Examples:
    ```python
        # apply a 3x3 unshared weights convolution with 64 output filters on a
        32x32 image
        # with `data_format="channels_last"`:
        model = Sequential()
        model.add(LocallyConnected2D(64, (3, 3), input_shape=(32, 32, 3)))
        # now model.output_shape == (None, 30, 30, 64)
        # notice that this layer will consume (30*30)*(3*3*3*64) + (30*30)*64
        parameters

        # add a 3x3 unshared weights convolution on top, with 32 output filters:
        model.add(LocallyConnected2D(32, (3, 3)))
        # now model.output_shape == (None, 28, 28, 32)
    ```

    Args:
        filters: Integer, the dimensionality of the output space (i.e. the
          number of output filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
          width and height of the 2D convolution window. Can be a single integer
          to specify the same value for all spatial dimensions.
        strides: An integer or tuple/list of 2 integers, specifying the strides
          of the convolution along the width and height. Can be a single integer
          to specify the same value for all spatial dimensions.
        padding: Currently only support `"valid"` (case-insensitive). `"same"`
          will be supported in future. `"valid"` means no padding.
        data_format: A string, one of `channels_last` (default) or
          `channels_first`. The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape `(batch, height,
            width, channels)` while `channels_first` corresponds to inputs with
            shape
          `(batch, channels, height, width)`. When unspecified, uses
          `image_data_format` value found in your Keras config file at
          `~/.keras/keras.json` (if exists) else 'channels_last'.
          Defaults to 'channels_last'.
        activation: Activation function to use. If you don't specify anything,
          no activation is applied (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights
          matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to the output of the
          layer (its "activation").
        kernel_constraint: Constraint function applied to the kernel matrix.
        bias_constraint: Constraint function applied to the bias vector.
        implementation: implementation mode, either `1`, `2`, or `3`. `1` loops
          over input spatial locations to perform the forward pass. It is
          memory-efficient but performs a lot of (small) ops.  `2` stores layer
          weights in a dense but sparsely-populated 2D matrix and implements the
          forward pass as a single matrix-multiply. It uses a lot of RAM but
          performs few (large) ops.  `3` stores layer weights in a sparse tensor
          and implements the forward pass as a single sparse matrix-multiply.
            How to choose:
            `1`: large, dense models,
            `2`: small models,
            `3`: large, sparse models,  where "large" stands for large
              input/output activations (i.e. many `filters`, `input_filters`,
              large `np.prod(input_size)`, `np.prod(output_size)`), and "sparse"
              stands for few connections between inputs and outputs, i.e. small
              ratio `filters * input_filters * np.prod(kernel_size) /
              (np.prod(input_size) * np.prod(strides))`, where inputs to and
              outputs of the layer are assumed to have shapes `input_size +
              (input_filters,)`, `output_size + (filters,)` respectively. It is
              recommended to benchmark each in the setting of interest to pick
              the most efficient one (in terms of speed and memory usage).
              Correct choice of implementation can lead to dramatic speed
              improvements (e.g. 50X), potentially at the expense of RAM. Also,
              only `padding="valid"` is supported by `implementation=1`.
    Input shape:
        4D tensor with shape: `(samples, channels, rows, cols)` if
          data_format='channels_first'
        or 4D tensor with shape: `(samples, rows, cols, channels)` if
          data_format='channels_last'.
    Output shape:
        4D tensor with shape: `(samples, filters, new_rows, new_cols)` if
          data_format='channels_first'
        or 4D tensor with shape: `(samples, new_rows, new_cols, filters)` if
          data_format='channels_last'. `rows` and `cols` values might have
          changed due to padding.
    