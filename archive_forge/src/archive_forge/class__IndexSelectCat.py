from typing import Optional, Sequence
import torch
from xformers.ops._triton import (
from .common import BaseOperator, register_operator
class _IndexSelectCat(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args: torch.Tensor) -> torch.Tensor:
        assert len(args) % 2 == 0
        sources = args[:len(args) // 2]
        indices = args[len(args) // 2:]
        output_numel = 0
        for source, index in zip(sources, indices):
            num_rows, num_cols = source.shape
            num_indices = index.shape[0]
            output_numel += num_indices * num_cols
        output = torch.empty([output_numel], dtype=sources[0].dtype, device=sources[0].device)
        processed_numel = 0
        for source, index in zip(sources, indices):
            num_indices = index.shape[0]
            num_cols = source.shape[1]
            if index_select_cat_fwd is not None:
                index_select_cat_fwd(output[processed_numel:processed_numel + num_indices * num_cols].view([num_indices, num_cols]), source, index)
            else:
                raise RuntimeError('Triton is needed for forward pass but it is not available!')
            processed_numel += num_indices * num_cols
        ctx.save_for_backward(*indices)
        ctx.source_shapes = [source.shape for source in sources]
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        indices = ctx.saved_tensors
        gradients = []
        processed_numel = 0
        for source_shape, index in zip(ctx.source_shapes, indices):
            num_rows, num_cols = source_shape
            num_indices = index.shape[0]
            grad_output_slice = grad_output[processed_numel:processed_numel + num_indices * num_cols].reshape([num_indices, num_cols])
            processed_numel += num_indices * num_cols
            grad_source_slice = torch.zeros([num_rows, num_cols], dtype=grad_output.dtype, device=grad_output.device)
            if index_select_cat_bwd is not None:
                index_select_cat_bwd(grad_source_slice, index, grad_output_slice)
            else:
                raise RuntimeError('Triton is needed for backward pass but it is not available!')
            gradients.append(grad_source_slice)
        return (*gradients, *[None] * len(gradients))