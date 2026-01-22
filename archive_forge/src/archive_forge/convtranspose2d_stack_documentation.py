from typing import Tuple
from ray.rllib.models.torch.misc import Reshape
from ray.rllib.models.utils import get_activation_fn, get_initializer
from ray.rllib.utils.framework import try_import_torch
Initializes a TransposedConv2DStack instance.

        Args:
            input_size: The size of the 1D input vector, from which to
                generate the image distribution.
            filters (Tuple[Tuple[int]]): Tuple of filter setups (1 for each
                ConvTranspose2D layer): [in_channels, kernel, stride].
            initializer (Union[str]):
            bias_init: The initial bias values to use.
            activation_fn: Activation function descriptor (str).
            output_shape (Tuple[int]): Shape of the final output image.
        