import torch
from functools import partial
from torch.testing import make_tensor
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.common_dtype import all_types_and
import numpy as np
class ForwardHasDefaultArgs(torch.autograd.Function):

    @staticmethod
    def forward(x, idx=(2,)):
        return x[idx]

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, idx = inputs
        ctx.x_shape = x.shape
        ctx.idx = idx

    @staticmethod
    def backward(ctx, grad_output):
        result = grad_output.new_zeros(ctx.x_shape)
        result[ctx.idx] = grad_output
        return (result, None)

    @staticmethod
    def vmap(info, in_dims, x, idx):
        x_bdim, _ = in_dims
        x = x.movedim(x_bdim, 1)
        return (ForwardHasDefaultArgs.apply(x, idx), 0)

    @staticmethod
    def jvp(ctx, x_tangent, _):
        return ForwardHasDefaultArgs.apply(x_tangent, ctx.idx)