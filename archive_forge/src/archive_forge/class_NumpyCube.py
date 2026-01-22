import torch
from functools import partial
from torch.testing import make_tensor
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.common_dtype import all_types_and
import numpy as np
class NumpyCube(torch.autograd.Function):

    @staticmethod
    def forward(input):
        input_np = to_numpy(input)
        dinput = torch.tensor(3 * input_np ** 2, device=input.device)
        return (torch.tensor(input_np ** 3, device=input.device), dinput)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[0], output[1])
        ctx.save_for_forward(inputs[0], output[1])

    @staticmethod
    def backward(ctx, grad_output, grad_saved):
        input, dinput = ctx.saved_tensors
        return NumpyMul.apply(grad_output, dinput) + 6 * NumpyMul.apply(grad_saved, input)

    @staticmethod
    def vmap(info, in_dims, input):
        result = NumpyCube.apply(input)
        return (result, (in_dims[0], in_dims[0]))

    @staticmethod
    def jvp(ctx, input_tangent):
        input, dinput = ctx.saved_tensors
        return (NumpyMul.apply(input_tangent, dinput), 6 * NumpyMul.apply(input_tangent, input))