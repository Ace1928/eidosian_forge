import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.distributed import group, ReduceOp
class _Broadcast(Function):

    @staticmethod
    def forward(ctx, src, group, tensor):
        ctx.src = src
        ctx.group = group
        ctx.rank = dist.get_rank()
        tensor = tensor.clone()
        dist.broadcast(tensor, src, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        gx = _Reduce.apply(ctx.src, ReduceOp.SUM, ctx.group, grad_output)
        if ctx.src != ctx.rank:
            gx.zero_()
        return (None, None, gx)