import torch
from functools import partial
from torch.testing import make_tensor
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.common_dtype import all_types_and
import numpy as np
class SortGenVmap(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(x, dim):
        device = x.device
        ind = torch.argsort(x, dim=dim)
        ind_inv = torch.argsort(ind, axis=dim)
        result = torch.take_along_dim(x, ind, dim=dim)
        return (result, ind, ind_inv)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        x, dim = inputs
        _, ind, ind_inv = outputs
        ctx.mark_non_differentiable(ind, ind_inv)
        ctx.save_for_backward(ind, ind_inv)
        ctx.save_for_forward(ind, ind_inv)
        ctx.dim = dim

    @staticmethod
    def backward(ctx, grad_output, _0, _1):
        ind, ind_inv = ctx.saved_tensors
        return (TakeGenVmap.apply(grad_output, ind_inv, ind, ctx.dim), None)

    @staticmethod
    def jvp(ctx, x_tangent, _):
        ind, ind_inv = ctx.saved_tensors
        return (TakeGenVmap.apply(x_tangent, ind, ind_inv, ctx.dim), None, None)