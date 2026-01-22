from typing import Any, Optional, Tuple
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
class GetSumFunction(torch.autograd.Function):
    """Custom checkpointed function to get sum-per-token from an input and a weight."""

    @staticmethod
    def get_sum(i: torch.Tensor, w: torch.Tensor, tgt: torch.Tensor, maxs: torch.Tensor, w_idx: int, full_precision: bool, margin: float, scale: Optional[float]) -> torch.Tensor:
        if scale is not None:
            _s = lmcl_matmul(i, w, tgt, w_idx, margin, scale)
        else:
            _s = torch.matmul(i, w.T)
        if full_precision:
            _s = _s.float()
        _s = (_s - maxs.reshape(-1, 1)).exp().sum(dim=1)
        return _s

    @staticmethod
    def forward(ctx: Any, i: torch.Tensor, w: torch.Tensor, tgt: torch.Tensor, maxs: torch.Tensor, kernel_obj: 'MemoryEfficientVocabOutput', w_idx: int, w_split_size: int, split_dim: int) -> torch.Tensor:
        """Forward function that computes the sum, without saving activations."""
        if DEBUG and dist.is_initialized() and (dist.get_rank() == 0):
            print('DEBUG sum fwd')
        ctx.save_for_backward(i, w, tgt, maxs)
        ctx.kernel_obj = kernel_obj
        ctx.w_idx = w_idx
        ctx.w_split_size = w_split_size
        assert split_dim == 0
        with torch.no_grad():
            return GetSumFunction.get_sum(i, w, tgt, maxs, w_idx, kernel_obj.fp_sum, kernel_obj.margin, kernel_obj.scale)

    @staticmethod
    def backward(ctx: Any, *args: Any) -> Any:
        """Recompute the forward sum and backward grad.

        Accumulate the grad to the right split of the full grad.
        """
        if DEBUG and dist.is_initialized() and (dist.get_rank() == 0):
            print('DEBUG sum bwd')
        assert len(args) == 1
        assert ctx.kernel_obj.proj_weight.grad is not None
        i, w, tgt, maxs = ctx.saved_tensors
        assert i.requires_grad
        assert w.requires_grad
        assert maxs.requires_grad
        i = i.detach().requires_grad_(True)
        w = w.detach().requires_grad_(True)
        maxs = maxs.detach().requires_grad_(True)
        with torch.enable_grad():
            sums = GetSumFunction.get_sum(i, w, tgt, maxs, ctx.w_idx, ctx.kernel_obj.fp_sum, ctx.kernel_obj.margin, ctx.kernel_obj.scale)
        torch.autograd.backward(sums, *args)
        assert w.grad is not None
        with torch.no_grad():
            grads = torch.split(ctx.kernel_obj.proj_weight.grad, ctx.w_split_size)
            grads[ctx.w_idx].add_(w.grad)
        return (i.grad, None, None, maxs.grad, None, None, None, None)