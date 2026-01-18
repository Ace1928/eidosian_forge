from keras.src import backend
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.engine.base_layer import Layer
from keras.src.engine.input_spec import InputSpec
from keras.src.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export
Parametric Rectified Linear Unit.

    It follows:

    ```
        f(x) = alpha * x for x < 0
        f(x) = x for x >= 0
    ```

    where `alpha` is a learned array with the same shape as x.

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as the input.

    Args:
        alpha_initializer: Initializer function for the weights.
        alpha_regularizer: Regularizer for the weights.
        alpha_constraint: Constraint for the weights.
        shared_axes: The axes along which to share learnable
            parameters for the activation function.
            For example, if the incoming feature maps
            are from a 2D convolution
            with output shape `(batch, height, width, channels)`,
            and you wish to share parameters across space
            so that each filter only has one set of parameters,
            set `shared_axes=[1, 2]`.
    