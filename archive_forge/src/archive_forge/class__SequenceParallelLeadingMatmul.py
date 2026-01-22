from typing import Callable, List, Optional, Tuple
import torch
from .common import make_pytorch_cuda_operator
from .differentiable_collectives import (
from .sequence_parallel_fused_ops import (
from .tiled_matmul import tiled_matmul_fwd
class _SequenceParallelLeadingMatmul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, fuse: bool, process_group: torch.distributed.ProcessGroup, scattered_input: torch.Tensor, *weights: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        ctx.save_for_backward(scattered_input, *weights)
        ctx.fuse = fuse
        ctx.process_group = process_group
        gathered_output = sequence_parallel_leading_matmul_fwd(scattered_input, list(weights), fuse, process_group)
        return tuple(gathered_output)

    @staticmethod
    def backward(ctx, *grad_gathered_outputs: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        scattered_input, *weights = ctx.saved_tensors
        grad_scattered_input, grad_weights = sequence_parallel_leading_matmul_bwd(scattered_input, list(weights), list(grad_gathered_outputs), ctx.fuse, ctx.process_group)
        return (None, None, grad_scattered_input, *grad_weights)