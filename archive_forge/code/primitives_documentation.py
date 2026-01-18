from typing import Callable, List, Optional, Union, Tuple
from ray.rllib.core.models.torch.utils import Stride2D
from ray.rllib.models.torch.misc import (
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.framework import try_import_torch
Initializes a TorchCNNTranspose instance.

        Args:
            input_dims: The 3D input dimensions of the network (incoming image).
            cnn_transpose_filter_specifiers: A list of lists, where each item represents
                one Conv2DTranspose layer. Each such Conv2DTranspose layer is further
                specified by the elements of the inner lists. The inner lists follow
                the format: `[number of filters, kernel, stride]` to
                specify a convolutional-transpose layer stacked in order of the
                outer list.
                `kernel` as well as `stride` might be provided as width x height tuples
                OR as single ints representing both dimension (width and height)
                in case of square shapes.
            cnn_transpose_use_bias: Whether to use bias on all Conv2DTranspose layers.
            cnn_transpose_use_layernorm: Whether to insert a LayerNormalization
                functionality in between each Conv2DTranspose layer's outputs and its
                activation.
                The last Conv2DTranspose layer will not be normed, regardless.
            cnn_transpose_activation: The activation function to use after each layer
                (except for the last Conv2DTranspose layer, which is always
                non-activated).
        