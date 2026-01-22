from typing import Callable, List, Optional, Tuple, Union
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.framework import try_import_tf
class TfCNN(tf.keras.Model):
    """A model containing a CNN with N Conv2D layers.

    All layers share the same activation function, bias setup (use bias or not),
    and LayerNormalization setup (use layer normalization or not).

    Note that there is no flattening nor an additional dense layer at the end of the
    stack. The output of the network is a 3D tensor of dimensions
    [width x height x num output filters].
    """

    def __init__(self, *, input_dims: Union[List[int], Tuple[int]], cnn_filter_specifiers: List[List[Union[int, List]]], cnn_use_bias: bool=True, cnn_use_layernorm: bool=False, cnn_activation: Optional[str]='relu'):
        """Initializes a TfCNN instance.

        Args:
            input_dims: The 3D input dimensions of the network (incoming image).
            cnn_filter_specifiers: A list in which each element is another (inner) list
                of either the following forms:
                `[number of channels/filters, kernel, stride]`
                OR:
                `[number of channels/filters, kernel, stride, padding]`, where `padding`
                can either be "same" or "valid".
                When using the first format w/o the `padding` specifier, `padding` is
                "same" by default. Also, `kernel` and `stride` may be provided either as
                single ints (square) or as a tuple/list of two ints (width- and height
                dimensions) for non-squared kernel/stride shapes.
                A good rule of thumb for constructing CNN stacks is:
                When using padding="same", the input "image" will be reduced in size by
                the factor `stride`, e.g. input=(84, 84, 3) stride=2 kernel=x
                padding="same" filters=16 -> output=(42, 42, 16).
                For example, if you would like to reduce an Atari image from its
                original (84, 84, 3) dimensions down to (6, 6, F), you can construct the
                following stack and reduce the w x h dimension of the image by 2 in each
                layer:
                [[16, 4, 2], [32, 4, 2], [64, 4, 2], [128, 4, 2]] -> output=(6, 6, 128)
            cnn_use_bias: Whether to use bias on all Conv2D layers.
            cnn_activation: The activation function to use after each Conv2D layer.
            cnn_use_layernorm: Whether to insert a LayerNormalization functionality
                in between each Conv2D layer's outputs and its activation.
        """
        super().__init__()
        assert len(input_dims) == 3
        cnn_activation = get_activation_fn(cnn_activation, framework='tf2')
        layers = []
        layers.append(tf.keras.layers.Input(shape=input_dims))
        for filter_specs in cnn_filter_specifiers:
            if len(filter_specs) == 3:
                num_filters, kernel_size, strides = filter_specs
                padding = 'same'
            else:
                num_filters, kernel_size, strides, padding = filter_specs
            layers.append(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=cnn_use_bias, activation=None if cnn_use_layernorm else cnn_activation))
            if cnn_use_layernorm:
                layers.append(tf.keras.layers.LayerNormalization(axis=[-3, -2, -1], epsilon=1e-05))
                layers.append(tf.keras.layers.Activation(cnn_activation))
        self.cnn = tf.keras.Sequential(layers)
        self.expected_input_dtype = tf.float32

    def call(self, inputs, **kwargs):
        return self.cnn(tf.cast(inputs, self.expected_input_dtype))