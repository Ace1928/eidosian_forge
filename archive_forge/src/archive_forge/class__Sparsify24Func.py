import contextlib
import ctypes
import glob
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, TypeVar, cast
import torch
from .common import BaseOperator, get_operator, get_xformers_operator, register_operator
class _Sparsify24Func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, algo: str, gradient: str, backend: str):
        if gradient not in [GRADIENT_SP24, GRADIENT_DENSE]:
            raise ValueError(f"Invalid gradient type: '{gradient}'. Expected '{GRADIENT_SP24}' or '{GRADIENT_DENSE}'")
        if not isinstance(x, Sparse24Tensor):
            packed, meta, packed_t, meta_t, threads_masks = SparsifyBothWays.OPERATOR(x, algorithm=algo, backend=backend)
            cls = Sparse24TensorCutlass if backend == BACKEND_CUTLASS else Sparse24TensorCuSparseLt
            out = cls(x.shape, packed=packed, meta=meta, packed_t=packed_t, meta_t=meta_t, threads_masks=threads_masks, requires_grad=False)
        else:
            if x.threads_masks is None:
                raise ValueError('!!')
            out = x
        ctx.threads_masks = out.threads_masks
        ctx.meta = out.meta
        ctx.meta_t = out.meta_t
        ctx.dtype = out.dtype
        ctx.gradient = gradient
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        if isinstance(grad_out, Sparse24Tensor):
            return (grad_out, None, None, None)
        assert not isinstance(grad_out, Sparse24Tensor)
        assert grad_out.dtype == ctx.dtype
        if ctx.gradient == GRADIENT_SP24:
            packed, packed_t = SparsifyApply.OPERATOR(grad_out, ctx.threads_masks)
            grad_in: torch.Tensor = Sparse24TensorCutlass(grad_out.shape, packed, ctx.meta, packed_t, ctx.meta_t, ctx.threads_masks, requires_grad=grad_out.requires_grad)
        elif ctx.gradient == GRADIENT_DENSE:
            assert ctx.threads_masks.is_contiguous()
            grad_in = SparsifyApplyDenseOutput.OPERATOR(grad_out, ctx.threads_masks)
        else:
            assert False, f'Unsupported gradient type: {ctx.gradient}'
        return (grad_in, None, None, None)