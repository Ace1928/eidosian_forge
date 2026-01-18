import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from .expanded_weights_utils import \
def unfold3d(tensor, kernel_size, padding, stride, dilation):
    """
    Extract sliding local blocks from an batched input tensor.

    :class:`torch.nn.Unfold` only supports 4D inputs (batched image-like tensors).
    This method implements the same action for 5D inputs
    Args:
        tensor: An input tensor of shape ``(B, C, D, H, W)``.
        kernel_size: the size of the sliding blocks
        padding: implicit zero padding to be added on both sides of input
        stride: the stride of the sliding blocks in the input spatial dimensions
        dilation: the spacing between the kernel points.
    Returns:
        A tensor of shape ``(B, C * np.prod(kernel_size), L)``, where L - output spatial dimensions.
        See :class:`torch.nn.Unfold` for more details
    Example:
        >>> # xdoctest: +SKIP
        >>> B, C, D, H, W = 3, 4, 5, 6, 7
        >>> tensor = torch.arange(1, B * C * D * H * W + 1.).view(B, C, D, H, W)
        >>> unfold3d(tensor, kernel_size=2, padding=0, stride=1).shape
        torch.Size([3, 32, 120])
    """
    if len(tensor.shape) != 5:
        raise ValueError(f'Input tensor must be of the shape [B, C, D, H, W]. Got{tensor.shape}')
    if dilation != (1, 1, 1):
        raise NotImplementedError(f'dilation={dilation} not supported.')
    batch_size, channels, _, _, _ = tensor.shape
    tensor = F.pad(tensor, (padding[2], padding[2], padding[1], padding[1], padding[0], padding[0]))
    tensor = tensor.unfold(dimension=2, size=kernel_size[0], step=stride[0])
    tensor = tensor.unfold(dimension=3, size=kernel_size[1], step=stride[1])
    tensor = tensor.unfold(dimension=4, size=kernel_size[2], step=stride[2])
    tensor = tensor.permute(0, 2, 3, 4, 1, 5, 6, 7)
    tensor = tensor.reshape(batch_size, -1, channels * np.prod(kernel_size)).transpose(1, 2)
    return tensor