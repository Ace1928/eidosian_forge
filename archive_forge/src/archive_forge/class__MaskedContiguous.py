from functools import partial
import torch
from .binary import (
from .core import is_masked_tensor, MaskedTensor, _get_data, _masks_match, _maybe_get_mask
from .passthrough import (
from .reductions import (
from .unary import (
class _MaskedContiguous(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        if not is_masked_tensor(input):
            raise ValueError('MaskedContiguous forward: input must be a MaskedTensor.')
        if input.is_contiguous():
            return input
        data = input.get_data()
        mask = input.get_mask()
        return MaskedTensor(data.contiguous(), mask.contiguous())

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output