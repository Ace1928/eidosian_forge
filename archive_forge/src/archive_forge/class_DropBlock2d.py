import torch
import torch.fx
import torch.nn.functional as F
from torch import nn, Tensor
from ..utils import _log_api_usage_once
class DropBlock2d(nn.Module):
    """
    See :func:`drop_block2d`.
    """

    def __init__(self, p: float, block_size: int, inplace: bool=False, eps: float=1e-06) -> None:
        super().__init__()
        self.p = p
        self.block_size = block_size
        self.inplace = inplace
        self.eps = eps

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor): Input feature map on which some areas will be randomly
                dropped.
        Returns:
            Tensor: The tensor after DropBlock layer.
        """
        return drop_block2d(input, self.p, self.block_size, self.inplace, self.eps, self.training)

    def __repr__(self) -> str:
        s = f'{self.__class__.__name__}(p={self.p}, block_size={self.block_size}, inplace={self.inplace})'
        return s