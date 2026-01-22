import torch
from .utils import _csr_to_coo, _transpose_with_info
class _sddmm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b, row_indices, row_offsets, column_indices, _transp_info):
        out = _sddmm_func(a, b, row_indices, row_offsets, column_indices)
        ctx.save_for_backward(a, b, row_indices, row_offsets, column_indices, *_transp_info)
        return out

    @staticmethod
    def backward(ctx, grad):
        a, b, row_indices, row_offsets, column_indices, *_transp_info = ctx.saved_tensors
        m, n = (a.shape[1], b.shape[1])
        grad = grad.contiguous()
        a = a.contiguous()
        b = b.contiguous()
        a_grad = torch.ops.xformers.spmm_sputnik(b, row_indices, grad, row_offsets, column_indices, m)
        row_indices_t, grad_t, row_offsets_t, column_indices_t = _transpose_with_info(grad, _transp_info)
        b_grad = torch.ops.xformers.spmm_sputnik(a, row_indices_t, grad_t, row_offsets_t, column_indices_t, n)
        return (a_grad, b_grad, None, None, None, None)