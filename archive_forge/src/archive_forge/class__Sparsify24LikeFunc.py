import contextlib
import ctypes
import glob
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, TypeVar, cast
import torch
from .common import BaseOperator, get_operator, get_xformers_operator, register_operator
class _Sparsify24LikeFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, pattern: Sparse24Tensor, out_dense: bool):
        assert isinstance(pattern, Sparse24Tensor)
        if not isinstance(pattern, Sparse24TensorCutlass):
            raise NotImplementedError('`sparsify24_like(x, pattern)` is only implemented for CUTLASS backend')
        if not pattern.threads_masks.is_contiguous():
            raise NotImplementedError('`sparsify24_like(x, pattern)` is not implemented when `pattern` is transposed')
        ctx.threads_masks = pattern.threads_masks
        ctx.meta = pattern.meta
        ctx.meta_t = pattern.meta_t
        ctx.dtype = pattern.dtype
        if out_dense:
            assert ctx.threads_masks.is_contiguous()
            return SparsifyApplyDenseOutput.OPERATOR(x, ctx.threads_masks)
        packed, packed_t = SparsifyApply.OPERATOR(x, ctx.threads_masks)
        return Sparse24TensorCutlass(x.shape, packed, ctx.meta, packed_t, ctx.meta_t, ctx.threads_masks, requires_grad=x.requires_grad)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        if isinstance(grad_out, Sparse24Tensor):
            return (grad_out, None, None)
        assert not isinstance(grad_out, Sparse24Tensor)
        assert grad_out.dtype == ctx.dtype
        packed, packed_t = SparsifyApply.OPERATOR(grad_out, ctx.threads_masks)
        return (Sparse24TensorCutlass(grad_out.shape, packed, ctx.meta, packed_t, ctx.meta_t, ctx.threads_masks, requires_grad=grad_out.requires_grad), None, None)