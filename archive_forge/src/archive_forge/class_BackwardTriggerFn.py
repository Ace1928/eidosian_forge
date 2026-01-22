from typing import Any, Optional, Tuple
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
class BackwardTriggerFn(torch.autograd.Function):
    """A backward trigger function."""

    @staticmethod
    def forward(ctx: Any, w: torch.Tensor, trigger_tensor: torch.Tensor) -> torch.Tensor:
        """We take a weight tensor and the trigger as inputs and output the weight directly."""
        if DEBUG and dist.is_initialized() and (dist.get_rank() == 0):
            print('DEBUG trigger fwd')
        ctx.save_for_backward(w, trigger_tensor)
        return w

    @staticmethod
    def backward(ctx: Any, *args: Any) -> Any:
        """We return zero grad for the trigger only."""
        if DEBUG and dist.is_initialized() and (dist.get_rank() == 0):
            print('DEBUG trigger bwd')
        assert len(args) == 1
        w, trigger = ctx.saved_tensors
        assert w.requires_grad
        assert trigger.requires_grad
        return (None, torch.zeros_like(trigger))