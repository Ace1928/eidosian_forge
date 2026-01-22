import math
import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import _standard_normal, lazy_property
def _batch_mahalanobis(bL, bx):
    """
    Computes the squared Mahalanobis distance :math:`\\mathbf{x}^\\top\\mathbf{M}^{-1}\\mathbf{x}`
    for a factored :math:`\\mathbf{M} = \\mathbf{L}\\mathbf{L}^\\top`.

    Accepts batches for both bL and bx. They are not necessarily assumed to have the same batch
    shape, but `bL` one should be able to broadcasted to `bx` one.
    """
    n = bx.size(-1)
    bx_batch_shape = bx.shape[:-1]
    bx_batch_dims = len(bx_batch_shape)
    bL_batch_dims = bL.dim() - 2
    outer_batch_dims = bx_batch_dims - bL_batch_dims
    old_batch_dims = outer_batch_dims + bL_batch_dims
    new_batch_dims = outer_batch_dims + 2 * bL_batch_dims
    bx_new_shape = bx.shape[:outer_batch_dims]
    for sL, sx in zip(bL.shape[:-2], bx.shape[outer_batch_dims:-1]):
        bx_new_shape += (sx // sL, sL)
    bx_new_shape += (n,)
    bx = bx.reshape(bx_new_shape)
    permute_dims = list(range(outer_batch_dims)) + list(range(outer_batch_dims, new_batch_dims, 2)) + list(range(outer_batch_dims + 1, new_batch_dims, 2)) + [new_batch_dims]
    bx = bx.permute(permute_dims)
    flat_L = bL.reshape(-1, n, n)
    flat_x = bx.reshape(-1, flat_L.size(0), n)
    flat_x_swap = flat_x.permute(1, 2, 0)
    M_swap = torch.linalg.solve_triangular(flat_L, flat_x_swap, upper=False).pow(2).sum(-2)
    M = M_swap.t()
    permuted_M = M.reshape(bx.shape[:-1])
    permute_inv_dims = list(range(outer_batch_dims))
    for i in range(bL_batch_dims):
        permute_inv_dims += [outer_batch_dims + i, old_batch_dims + i]
    reshaped_M = permuted_M.permute(permute_inv_dims)
    return reshaped_M.reshape(bx_batch_shape)