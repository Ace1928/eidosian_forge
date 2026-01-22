from typing import Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
def _uniform_filter(inputs: Tensor, window_size: int) -> Tensor:
    """Apply uniform filter with a window of a given size over the input image.

    Args:
        inputs: Input image
        window_size: Sliding window used for rmse calculation

    Return:
        Image transformed with the uniform input

    """
    inputs = _reflection_pad_2d(inputs, window_size // 2, window_size % 2)
    kernel_weight, kernel_bias = _uniform_weight_bias_conv2d(inputs, window_size)
    return torch.cat([F.conv2d(inputs[:, channel].unsqueeze(1), kernel_weight, kernel_bias, padding=0) for channel in range(inputs.shape[1])], dim=1)