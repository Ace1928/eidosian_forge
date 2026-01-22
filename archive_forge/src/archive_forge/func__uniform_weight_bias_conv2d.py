from typing import Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
def _uniform_weight_bias_conv2d(inputs: Tensor, window_size: int) -> Tuple[Tensor, Tensor]:
    """Construct uniform weight and bias for a 2d convolution.

    Args:
        inputs: Input image
        window_size: size of convolutional kernel

    Return:
        The weight and bias for 2d convolution

    """
    kernel_weight = torch.ones(1, 1, window_size, window_size, dtype=inputs.dtype, device=inputs.device)
    kernel_weight /= window_size ** 2
    kernel_bias = torch.zeros(1, dtype=inputs.dtype, device=inputs.device)
    return (kernel_weight, kernel_bias)