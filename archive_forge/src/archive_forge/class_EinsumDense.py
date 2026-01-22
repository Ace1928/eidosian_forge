import re
import tensorflow.compat.v2 as tf
from keras.src import activations
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.engine.base_layer import Layer
from tensorflow.python.util.tf_export import keras_export
@keras_export('keras.layers.EinsumDense', 'keras.layers.experimental.EinsumDense')
class EinsumDense(Layer):
    """A layer that uses `tf.einsum` as the backing computation.

    This layer can perform einsum calculations of arbitrary dimensionality.

    Args:
      equation: An equation describing the einsum to perform. This equation must
        be a valid einsum string of the form `ab,bc->ac`, `...ab,bc->...ac`, or
        `ab...,bc->ac...` where 'ab', 'bc', and 'ac' can be any valid einsum
        axis expression sequence.
      output_shape: The expected shape of the output tensor (excluding the batch
        dimension and any dimensions represented by ellipses). You can specify
        None for any dimension that is unknown or can be inferred from the input
        shape.
      activation: Activation function to use. If you don't specify anything, no
        activation is applied (that is, a "linear" activation: `a(x) = x`).
      bias_axes: A string containing the output dimension(s) to apply a bias to.
        Each character in the `bias_axes` string should correspond to a
        character in the output portion of the `equation` string.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to the output of the
        layer (its "activation").
      kernel_constraint: Constraint function applied to the `kernel` weights
        matrix.
      bias_constraint: Constraint function applied to the bias vector.

    Examples:

    **Biased dense layer with einsums**

    This example shows how to instantiate a standard Keras dense layer using
    einsum operations. This example is equivalent to
    `tf.keras.layers.Dense(64, use_bias=True)`.

    >>> layer = tf.keras.layers.EinsumDense("ab,bc->ac",
    ...                                     output_shape=64,
    ...                                     bias_axes="c")
    >>> input_tensor = tf.keras.Input(shape=[32])
    >>> output_tensor = layer(input_tensor)
    >>> output_tensor
    <... shape=(None, 64) dtype=...>

    **Applying a dense layer to a sequence**

    This example shows how to instantiate a layer that applies the same dense
    operation to every element in a sequence. Here, the `output_shape` has two
    values (since there are two non-batch dimensions in the output); the first
    dimension in the `output_shape` is `None`, because the sequence dimension
    `b` has an unknown shape.

    >>> layer = tf.keras.layers.EinsumDense("abc,cd->abd",
    ...                                     output_shape=(None, 64),
    ...                                     bias_axes="d")
    >>> input_tensor = tf.keras.Input(shape=[32, 128])
    >>> output_tensor = layer(input_tensor)
    >>> output_tensor
    <... shape=(None, 32, 64) dtype=...>

    **Applying a dense layer to a sequence using ellipses**

    This example shows how to instantiate a layer that applies the same dense
    operation to every element in a sequence, but uses the ellipsis notation
    instead of specifying the batch and sequence dimensions.

    Because we are using ellipsis notation and have specified only one axis, the
    `output_shape` arg is a single value. When instantiated in this way, the
    layer can handle any number of sequence dimensions - including the case
    where no sequence dimension exists.

    >>> layer = tf.keras.layers.EinsumDense("...x,xy->...y",
    ...                                     output_shape=64,
    ...                                     bias_axes="y")
    >>> input_tensor = tf.keras.Input(shape=[32, 128])
    >>> output_tensor = layer(input_tensor)
    >>> output_tensor
    <... shape=(None, 32, 64) dtype=...>
    """

    def __init__(self, equation, output_shape, activation=None, bias_axes=None, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs):
        super().__init__(**kwargs)
        self.equation = equation
        if isinstance(output_shape, int):
            self.partial_output_shape = [output_shape]
        else:
            self.partial_output_shape = list(output_shape)
        self.bias_axes = bias_axes
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        shape_data = _analyze_einsum_string(self.equation, self.bias_axes, input_shape, self.partial_output_shape)
        kernel_shape, bias_shape, self.full_output_shape = shape_data
        self.kernel = self.add_weight('kernel', shape=kernel_shape, initializer=self.kernel_initializer, regularizer=self.kernel_regularizer, constraint=self.kernel_constraint, dtype=self.dtype, trainable=True)
        if bias_shape is not None:
            self.bias = self.add_weight('bias', shape=bias_shape, initializer=self.bias_initializer, regularizer=self.bias_regularizer, constraint=self.bias_constraint, dtype=self.dtype, trainable=True)
        else:
            self.bias = None
        super().build(input_shape)

    def compute_output_shape(self, _):
        return tf.TensorShape(self.full_output_shape)

    def get_config(self):
        config = {'output_shape': self.partial_output_shape, 'equation': self.equation, 'activation': activations.serialize(self.activation), 'bias_axes': self.bias_axes, 'kernel_initializer': initializers.serialize(self.kernel_initializer), 'bias_initializer': initializers.serialize(self.bias_initializer), 'kernel_regularizer': regularizers.serialize(self.kernel_regularizer), 'bias_regularizer': regularizers.serialize(self.bias_regularizer), 'activity_regularizer': regularizers.serialize(self.activity_regularizer), 'kernel_constraint': constraints.serialize(self.kernel_constraint), 'bias_constraint': constraints.serialize(self.bias_constraint)}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        ret = tf.einsum(self.equation, inputs, self.kernel)
        if self.bias is not None:
            ret += self.bias
        if self.activation is not None:
            ret = self.activation(ret)
        return ret