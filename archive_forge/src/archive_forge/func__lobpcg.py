from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
from . import _linalg_utils as _utils
from .overrides import handle_torch_function, has_torch_function
def _lobpcg(A: Tensor, k: Optional[int]=None, B: Optional[Tensor]=None, X: Optional[Tensor]=None, n: Optional[int]=None, iK: Optional[Tensor]=None, niter: Optional[int]=None, tol: Optional[float]=None, largest: Optional[bool]=None, method: Optional[str]=None, tracker: None=None, ortho_iparams: Optional[Dict[str, int]]=None, ortho_fparams: Optional[Dict[str, float]]=None, ortho_bparams: Optional[Dict[str, bool]]=None) -> Tuple[Tensor, Tensor]:
    assert A.shape[-2] == A.shape[-1], A.shape
    if B is not None:
        assert A.shape == B.shape, (A.shape, B.shape)
    dtype = _utils.get_floating_dtype(A)
    device = A.device
    if tol is None:
        feps = {torch.float32: 1.2e-07, torch.float64: 2.23e-16}[dtype]
        tol = feps ** 0.5
    m = A.shape[-1]
    k = (1 if X is None else X.shape[-1]) if k is None else k
    n = (k if n is None else n) if X is None else X.shape[-1]
    if m < 3 * n:
        raise ValueError(f'LPBPCG algorithm is not applicable when the number of A rows (={m}) is smaller than 3 x the number of requested eigenpairs (={n})')
    method = 'ortho' if method is None else method
    iparams = {'m': m, 'n': n, 'k': k, 'niter': 1000 if niter is None else niter}
    fparams = {'tol': tol}
    bparams = {'largest': True if largest is None else largest}
    if method == 'ortho':
        if ortho_iparams is not None:
            iparams.update(ortho_iparams)
        if ortho_fparams is not None:
            fparams.update(ortho_fparams)
        if ortho_bparams is not None:
            bparams.update(ortho_bparams)
        iparams['ortho_i_max'] = iparams.get('ortho_i_max', 3)
        iparams['ortho_j_max'] = iparams.get('ortho_j_max', 3)
        fparams['ortho_tol'] = fparams.get('ortho_tol', tol)
        fparams['ortho_tol_drop'] = fparams.get('ortho_tol_drop', tol)
        fparams['ortho_tol_replace'] = fparams.get('ortho_tol_replace', tol)
        bparams['ortho_use_drop'] = bparams.get('ortho_use_drop', False)
    if not torch.jit.is_scripting():
        LOBPCG.call_tracker = LOBPCG_call_tracker
    if len(A.shape) > 2:
        N = int(torch.prod(torch.tensor(A.shape[:-2])))
        bA = A.reshape((N,) + A.shape[-2:])
        bB = B.reshape((N,) + A.shape[-2:]) if B is not None else None
        bX = X.reshape((N,) + X.shape[-2:]) if X is not None else None
        bE = torch.empty((N, k), dtype=dtype, device=device)
        bXret = torch.empty((N, m, k), dtype=dtype, device=device)
        for i in range(N):
            A_ = bA[i]
            B_ = bB[i] if bB is not None else None
            X_ = torch.randn((m, n), dtype=dtype, device=device) if bX is None else bX[i]
            assert len(X_.shape) == 2 and X_.shape == (m, n), (X_.shape, (m, n))
            iparams['batch_index'] = i
            worker = LOBPCG(A_, B_, X_, iK, iparams, fparams, bparams, method, tracker)
            worker.run()
            bE[i] = worker.E[:k]
            bXret[i] = worker.X[:, :k]
        if not torch.jit.is_scripting():
            LOBPCG.call_tracker = LOBPCG_call_tracker_orig
        return (bE.reshape(A.shape[:-2] + (k,)), bXret.reshape(A.shape[:-2] + (m, k)))
    X = torch.randn((m, n), dtype=dtype, device=device) if X is None else X
    assert len(X.shape) == 2 and X.shape == (m, n), (X.shape, (m, n))
    worker = LOBPCG(A, B, X, iK, iparams, fparams, bparams, method, tracker)
    worker.run()
    if not torch.jit.is_scripting():
        LOBPCG.call_tracker = LOBPCG_call_tracker_orig
    return (worker.E[:k], worker.X[:, :k])