from typing import Optional, Sequence
import torch
from xformers.ops._triton import (
from .common import BaseOperator, register_operator
class _ScaledIndexAdd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, index: torch.Tensor, source: torch.Tensor, scaling: Optional[torch.Tensor], alpha: float) -> torch.Tensor:
        if scaled_index_add_fwd is not None:
            scaled_index_add_fwd(x, index, source, scaling, alpha)
        else:
            raise RuntimeError('Triton is needed for forward pass but it is not available!')
        ctx.mark_dirty(x)
        ctx.save_for_backward(index, scaling, source)
        ctx.source_shape = source.shape
        ctx.alpha = alpha
        return x

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        index, scaling, source = ctx.saved_tensors
        grad_source = torch.empty_like(source)
        grad_scaling = None if scaling is None else torch.empty(ctx.source_shape, dtype=scaling.dtype, device=scaling.device)
        if scaled_index_add_bwd is not None:
            scaled_index_add_bwd(grad_output, grad_source, grad_scaling, source, scaling, index, ctx.alpha)
        else:
            raise RuntimeError('Triton is needed for backward pass but it is not available!')
        return (grad_output, None, grad_source, grad_scaling, None)