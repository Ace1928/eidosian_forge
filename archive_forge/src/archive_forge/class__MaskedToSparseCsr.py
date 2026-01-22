from functools import partial
import torch
from .binary import (
from .core import is_masked_tensor, MaskedTensor, _get_data, _masks_match, _maybe_get_mask
from .passthrough import (
from .reductions import (
from .unary import (
class _MaskedToSparseCsr(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        if not is_masked_tensor(input):
            raise ValueError('MaskedToSparseCsr forward: input must be a MaskedTensor.')
        if input._masked_data.ndim != 2:
            raise ValueError(f'Only 2D tensors can be converted to the SparseCsr layout but got shape: {input._masked_data.size()}')
        if input.layout == torch.sparse_csr:
            return input
        data = input.get_data()
        mask = input.get_mask()
        sparse_mask = mask.to_sparse_csr()
        sparse_data = data.sparse_mask(sparse_mask)
        return MaskedTensor(sparse_data, sparse_mask)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.to_dense()