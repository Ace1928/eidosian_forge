import torch
from functools import partial
from torch.testing import make_tensor
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.common_dtype import all_types_and
import numpy as np
class NumpyCubeNotComposable(torch.autograd.Function):

    @staticmethod
    def forward(input):
        input_np = to_numpy(input)
        return (torch.tensor(input_np ** 3, device=input.device), input_np)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, input_np = output
        ctx.input_np = input_np
        ctx.device = inputs[0].device

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output, grad_saved):
        result_np = 3 * ctx.input_np ** 2
        return torch.tensor(result_np, device=ctx.device)