import torch
from functools import partial
from torch.testing import make_tensor
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.common_dtype import all_types_and
import numpy as np
class ZeroGradientsGenVmap(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(x, y):
        return (x.clone(), y.clone())

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def backward(ctx, gx, gy):
        return (torch.zeros(3, 4, *gx.shape, dtype=gx.dtype, device=gx.device), torch.zeros(gy.shape, dtype=gy.dtype, device=gy.device))

    @staticmethod
    def jvp(ctx, gx, gy):
        return (torch.zeros(gx.shape, dtype=gx.dtype, device=gx.device), torch.zeros(gy.shape, dtype=gy.dtype, device=gy.device))