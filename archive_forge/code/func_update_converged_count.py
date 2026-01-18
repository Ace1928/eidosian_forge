from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
from . import _linalg_utils as _utils
from .overrides import handle_torch_function, has_torch_function
def update_converged_count(self):
    """Determine the number of converged eigenpairs using backward stable
        convergence criterion, see discussion in Sec 4.3 of [DuerschEtal2018].

        Users may redefine this method for custom convergence criteria.
        """
    prev_count = self.ivars['converged_count']
    tol = self.fparams['tol']
    A_norm = self.fvars['A_norm']
    B_norm = self.fvars['B_norm']
    E, X, R = (self.E, self.X, self.R)
    rerr = torch.norm(R, 2, (0,)) * (torch.norm(X, 2, (0,)) * (A_norm + E[:X.shape[-1]] * B_norm)) ** (-1)
    converged = rerr < tol
    count = 0
    for b in converged:
        if not b:
            break
        count += 1
    assert count >= prev_count, f'the number of converged eigenpairs (was {prev_count}, got {count}) cannot decrease'
    self.ivars['converged_count'] = count
    self.tvars['rerr'] = rerr
    return count