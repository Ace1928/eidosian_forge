from typing import List, Optional, Sequence, Tuple, Union
import torch
from .common import _get_storage_base
class _Unbind(torch.autograd.Function):
    """
    See function `unbind`
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, dim: int):
        ctx.dim = dim
        return x.unbind(dim)

    @classmethod
    def backward(cls, ctx, *tensors: torch.Tensor):
        return (_stack_fw(tensors, ctx.dim), None)