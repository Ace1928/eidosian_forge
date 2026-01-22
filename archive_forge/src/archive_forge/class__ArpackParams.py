import numpy as np
import warnings
from scipy.sparse.linalg._interface import aslinearoperator, LinearOperator
from scipy.sparse import eye, issparse
from scipy.linalg import eig, eigh, lu_factor, lu_solve
from scipy.sparse._sputils import isdense, is_pydata_spmatrix
from scipy.sparse.linalg import gmres, splu
from scipy._lib._util import _aligned_zeros
from scipy._lib._threadsafety import ReentrancyLock
from . import _arpack
class _ArpackParams:

    def __init__(self, n, k, tp, mode=1, sigma=None, ncv=None, v0=None, maxiter=None, which='LM', tol=0):
        if k <= 0:
            raise ValueError('k must be positive, k=%d' % k)
        if maxiter is None:
            maxiter = n * 10
        if maxiter <= 0:
            raise ValueError('maxiter must be positive, maxiter=%d' % maxiter)
        if tp not in 'fdFD':
            raise ValueError("matrix type must be 'f', 'd', 'F', or 'D'")
        if v0 is not None:
            self.resid = np.array(v0, copy=True)
            info = 1
        else:
            self.resid = np.zeros(n, tp)
            info = 0
        if sigma is None:
            self.sigma = 0
        else:
            self.sigma = sigma
        if ncv is None:
            ncv = choose_ncv(k)
        ncv = min(ncv, n)
        self.v = np.zeros((n, ncv), tp)
        self.iparam = np.zeros(11, arpack_int)
        ishfts = 1
        self.mode = mode
        self.iparam[0] = ishfts
        self.iparam[2] = maxiter
        self.iparam[3] = 1
        self.iparam[6] = mode
        self.n = n
        self.tol = tol
        self.k = k
        self.maxiter = maxiter
        self.ncv = ncv
        self.which = which
        self.tp = tp
        self.info = info
        self.converged = False
        self.ido = 0

    def _raise_no_convergence(self):
        msg = 'No convergence (%d iterations, %d/%d eigenvectors converged)'
        k_ok = self.iparam[4]
        num_iter = self.iparam[2]
        try:
            ev, vec = self.extract(True)
        except ArpackError as err:
            msg = f'{msg} [{err}]'
            ev = np.zeros((0,))
            vec = np.zeros((self.n, 0))
            k_ok = 0
        raise ArpackNoConvergence(msg % (num_iter, k_ok, self.k), ev, vec)