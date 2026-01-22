from typing import Optional, Tuple
import torch
import torch.distributed
class _ScatterToSequenceParallelRegion(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, process_group: torch.distributed.ProcessGroup) -> torch.Tensor:
        ctx.process_group = process_group
        return reduce_scatter_along_first_dim(x, process_group=process_group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return (gather_along_first_dim(grad_output, process_group=ctx.process_group), None)