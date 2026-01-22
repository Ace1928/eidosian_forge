import torch
from functools import partial
from torch.testing import make_tensor
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.common_dtype import all_types_and
import numpy as np
class TakeGenVmap(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(x, ind, ind_inv, dim):
        return torch.take_along_dim(x, ind, dim)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        x, ind, ind_inv, dim = inputs
        ctx.save_for_backward(ind, ind_inv)
        ctx.save_for_forward(ind, ind_inv)
        ctx.dim = dim

    @staticmethod
    def backward(ctx, grad_output):
        ind, ind_inv = ctx.saved_tensors
        result = TakeGenVmap.apply(grad_output, ind_inv, ind, ctx.dim)
        return (result, None, None, None)

    @staticmethod
    def jvp(ctx, x_tangent, ind_tangent, ind_inv_tangent, _):
        ind, ind_inv = ctx.saved_tensors
        return TakeGenVmap.apply(x_tangent, ind, ind_inv, ctx.dim)