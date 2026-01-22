from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
from . import _linalg_utils as _utils
from .overrides import handle_torch_function, has_torch_function
def _symeig_backward(D_grad, U_grad, A, D, U, largest):
    if U.size(-1) == U.size(-2):
        return _symeig_backward_complete_eigenspace(D_grad, U_grad, A, D, U)
    else:
        return _symeig_backward_partial_eigenspace(D_grad, U_grad, A, D, U, largest)