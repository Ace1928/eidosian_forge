import torch
from functools import partial
from torch.testing import make_tensor
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.common_dtype import all_types_and
import numpy as np
class MulGenVmap(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(x, y):
        return x * y

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        ctx.save_for_backward(*inputs)
        ctx.save_for_forward(*inputs)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        gx = None
        if ctx.needs_input_grad[0]:
            gx = MulGenVmap.apply(grad_output, y)
        gy = None
        if ctx.needs_input_grad[1]:
            gy = MulGenVmap.apply(grad_output, x)
        return (gx, gy)

    @staticmethod
    def jvp(ctx, x_tangent, y_tangent):
        x, y = ctx.saved_tensors
        return x_tangent * y + y_tangent * x