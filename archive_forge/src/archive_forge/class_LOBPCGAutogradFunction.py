from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
from . import _linalg_utils as _utils
from .overrides import handle_torch_function, has_torch_function
class LOBPCGAutogradFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A: Tensor, k: Optional[int]=None, B: Optional[Tensor]=None, X: Optional[Tensor]=None, n: Optional[int]=None, iK: Optional[Tensor]=None, niter: Optional[int]=None, tol: Optional[float]=None, largest: Optional[bool]=None, method: Optional[str]=None, tracker: None=None, ortho_iparams: Optional[Dict[str, int]]=None, ortho_fparams: Optional[Dict[str, float]]=None, ortho_bparams: Optional[Dict[str, bool]]=None) -> Tuple[Tensor, Tensor]:
        A = A.contiguous() if not A.is_sparse else A
        if B is not None:
            B = B.contiguous() if not B.is_sparse else B
        D, U = _lobpcg(A, k, B, X, n, iK, niter, tol, largest, method, tracker, ortho_iparams, ortho_fparams, ortho_bparams)
        ctx.save_for_backward(A, B, D, U)
        ctx.largest = largest
        return (D, U)

    @staticmethod
    def backward(ctx, D_grad, U_grad):
        A_grad = B_grad = None
        grads = [None] * 14
        A, B, D, U = ctx.saved_tensors
        largest = ctx.largest
        if A.is_sparse or (B is not None and B.is_sparse and ctx.needs_input_grad[2]):
            raise ValueError('lobpcg.backward does not support sparse input yet.Note that lobpcg.forward does though.')
        if A.dtype in (torch.complex64, torch.complex128) or (B is not None and B.dtype in (torch.complex64, torch.complex128)):
            raise ValueError('lobpcg.backward does not support complex input yet.Note that lobpcg.forward does though.')
        if B is not None:
            raise ValueError('lobpcg.backward does not support backward with B != I yet.')
        if largest is None:
            largest = True
        if B is None:
            A_grad = _symeig_backward(D_grad, U_grad, A, D, U, largest)
        grads[0] = A_grad
        grads[2] = B_grad
        return tuple(grads)