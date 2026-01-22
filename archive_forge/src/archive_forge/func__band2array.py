import numpy as np
import numpy.linalg as L
from scipy.linalg import solveh_banded
from scipy.optimize import golden
from models import _hbspline     #removed because this was segfaulting
import warnings
def _band2array(a, lower=0, symmetric=False, hermitian=False):
    """
    Take an upper or lower triangular banded matrix and return a
    numpy array.

    INPUTS:
       a         -- a matrix in upper or lower triangular banded matrix
       lower     -- is the matrix upper or lower triangular?
       symmetric -- if True, return the original result plus its transpose
       hermitian -- if True (and symmetric False), return the original
                    result plus its conjugate transposed
    """
    n = a.shape[1]
    r = a.shape[0]
    _a = 0
    if not lower:
        for j in range(r):
            _b = np.diag(a[r - 1 - j], k=j)[j:n + j, j:n + j]
            _a += _b
            if symmetric and j > 0:
                _a += _b.T
            elif hermitian and j > 0:
                _a += _b.conjugate().T
    else:
        for j in range(r):
            _b = np.diag(a[j], k=j)[0:n, 0:n]
            _a += _b
            if symmetric and j > 0:
                _a += _b.T
            elif hermitian and j > 0:
                _a += _b.conjugate().T
        _a = _a.T
    return _a