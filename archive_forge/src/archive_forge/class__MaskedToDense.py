from functools import partial
import torch
from .binary import (
from .core import is_masked_tensor, MaskedTensor, _get_data, _masks_match, _maybe_get_mask
from .passthrough import (
from .reductions import (
from .unary import (
class _MaskedToDense(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        if not is_masked_tensor(input):
            raise ValueError('MaskedToDense forward: input must be a MaskedTensor.')
        if input.layout == torch.strided:
            return input
        ctx.layout = input.layout
        data = input.get_data()
        mask = input.get_mask()
        return MaskedTensor(data.to_dense(), mask.to_dense())

    @staticmethod
    def backward(ctx, grad_output):
        layout = ctx.layout
        if layout == torch.sparse_coo:
            return grad_output.to_sparse_coo()
        elif layout == torch.sparse_csr:
            return grad_output.to_sparse_csr()
        elif layout == torch.strided:
            return grad_output.to_dense()
        raise ValueError('to_dense: Unsupported input layout: ', layout)