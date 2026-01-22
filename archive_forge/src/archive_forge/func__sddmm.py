import logging
import torch
from xformers import _is_triton_available
from xformers.ops import masked_matmul
def _sddmm(a, b, layout):
    block_size = a.shape[-2] // layout.shape[-2]
    a = a.reshape(a.shape[0], a.shape[1], a.shape[2] // block_size, block_size, a.shape[3])
    b = b.reshape(b.shape[0], b.shape[1], b.shape[2] // block_size, block_size, b.shape[3])
    h, r, c = layout.nonzero(as_tuple=True)
    out = torch.einsum('nhik,nhjk->nhij', a[:, h, r, :, :], b[:, h, c, :, :])
    return out