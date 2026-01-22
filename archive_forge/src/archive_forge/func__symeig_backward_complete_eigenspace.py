from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
from . import _linalg_utils as _utils
from .overrides import handle_torch_function, has_torch_function
def _symeig_backward_complete_eigenspace(D_grad, U_grad, A, D, U):
    F = D.unsqueeze(-2) - D.unsqueeze(-1)
    F.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))
    F.pow_(-1)
    Ut = U.mT.contiguous()
    res = torch.matmul(U, torch.matmul(torch.diag_embed(D_grad) + torch.matmul(Ut, U_grad) * F, Ut))
    return res