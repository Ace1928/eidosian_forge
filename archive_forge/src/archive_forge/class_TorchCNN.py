from typing import Callable, List, Optional, Union, Tuple
from ray.rllib.core.models.torch.utils import Stride2D
from ray.rllib.models.torch.misc import (
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.framework import try_import_torch
class TorchCNN(nn.Module):
    """A model containing a CNN with N Conv2D layers.

    All layers share the same activation function, bias setup (use bias or not),
    and LayerNorm setup (use layer normalization or not).

    Note that there is no flattening nor an additional dense layer at the end of the
    stack. The output of the network is a 3D tensor of dimensions
    [width x height x num output filters].
    """

    def __init__(self, *, input_dims: Union[List[int], Tuple[int]], cnn_filter_specifiers: List[List[Union[int, List]]], cnn_use_bias: bool=True, cnn_use_layernorm: bool=False, cnn_activation: str='relu'):
        """Initializes a TorchCNN instance.

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
        cnn_activation = get_activation_fn(cnn_activation, framework='torch')
        layers = []
        width, height, in_depth = input_dims
        in_size = [width, height]
        for filter_specs in cnn_filter_specifiers:
            if len(filter_specs) == 3:
                out_depth, kernel_size, strides = filter_specs
                padding = 'same'
            else:
                out_depth, kernel_size, strides, padding = filter_specs
            if padding == 'same':
                padding_size, out_size = same_padding(in_size, kernel_size, strides)
                layers.append(nn.ZeroPad2d(padding_size))
            else:
                out_size = valid_padding(in_size, kernel_size, strides)
            layers.append(nn.Conv2d(in_depth, out_depth, kernel_size, strides, bias=cnn_use_bias))
            if cnn_use_layernorm:
                layers.append(nn.LayerNorm((out_depth, out_size[0], out_size[1])))
            if cnn_activation is not None:
                layers.append(cnn_activation())
            in_size = out_size
            in_depth = out_depth
        self.cnn = nn.Sequential(*layers)
        self.expected_input_dtype = torch.float32

    def forward(self, inputs):
        inputs = inputs.permute(0, 3, 1, 2)
        out = self.cnn(inputs.type(self.expected_input_dtype))
        return out.permute(0, 2, 3, 1)