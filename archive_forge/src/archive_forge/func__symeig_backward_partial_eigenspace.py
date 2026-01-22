from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
from . import _linalg_utils as _utils
from .overrides import handle_torch_function, has_torch_function
def _symeig_backward_partial_eigenspace(D_grad, U_grad, A, D, U, largest):
    Ut = U.mT.contiguous()
    proj_U_ortho = -U.matmul(Ut)
    proj_U_ortho.diagonal(dim1=-2, dim2=-1).add_(1)
    gen = torch.Generator(A.device)
    U_ortho = proj_U_ortho.matmul(torch.randn((*A.shape[:-1], A.size(-1) - D.size(-1)), dtype=A.dtype, device=A.device, generator=gen))
    U_ortho_t = U_ortho.mT.contiguous()
    chr_poly_D = _polynomial_coefficients_given_roots(D)
    U_grad_projected = U_grad
    series_acc = U_grad_projected.new_zeros(U_grad_projected.shape)
    for k in range(1, chr_poly_D.size(-1)):
        poly_D = _vector_polynomial_value(chr_poly_D[..., k:], D)
        series_acc += U_grad_projected * poly_D.unsqueeze(-2)
        U_grad_projected = A.matmul(U_grad_projected)
    chr_poly_D_at_A = _matrix_polynomial_value(chr_poly_D, A)
    chr_poly_D_at_A_to_U_ortho = torch.matmul(U_ortho_t, torch.matmul(chr_poly_D_at_A, U_ortho))
    chr_poly_D_at_A_to_U_ortho_sign = -1 if largest and k % 2 == 1 else +1
    chr_poly_D_at_A_to_U_ortho_L = torch.linalg.cholesky(chr_poly_D_at_A_to_U_ortho_sign * chr_poly_D_at_A_to_U_ortho)
    res = _symeig_backward_complete_eigenspace(D_grad, U_grad, A, D, U)
    res -= U_ortho.matmul(chr_poly_D_at_A_to_U_ortho_sign * torch.cholesky_solve(U_ortho_t.matmul(series_acc), chr_poly_D_at_A_to_U_ortho_L)).matmul(Ut)
    return res