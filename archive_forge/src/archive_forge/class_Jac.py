import sys
import numpy as np
from scipy.linalg import norm, solve, inv, qr, svd, LinAlgError
from numpy import asarray, dot, vdot
import scipy.sparse.linalg
import scipy.sparse
from scipy.linalg import get_blas_funcs
import inspect
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from ._linesearch import scalar_search_wolfe1, scalar_search_armijo
class Jac(Jacobian):

    def update(self, x, F):
        self.x = x

    def solve(self, v, tol=0):
        m = J(self.x)
        if isinstance(m, np.ndarray):
            return solve(m, v)
        elif scipy.sparse.issparse(m):
            return spsolve(m, v)
        else:
            raise ValueError('Unknown matrix type')

    def matvec(self, v):
        m = J(self.x)
        if isinstance(m, np.ndarray):
            return dot(m, v)
        elif scipy.sparse.issparse(m):
            return m @ v
        else:
            raise ValueError('Unknown matrix type')

    def rsolve(self, v, tol=0):
        m = J(self.x)
        if isinstance(m, np.ndarray):
            return solve(m.conj().T, v)
        elif scipy.sparse.issparse(m):
            return spsolve(m.conj().T, v)
        else:
            raise ValueError('Unknown matrix type')

    def rmatvec(self, v):
        m = J(self.x)
        if isinstance(m, np.ndarray):
            return dot(m.conj().T, v)
        elif scipy.sparse.issparse(m):
            return m.conj().T @ v
        else:
            raise ValueError('Unknown matrix type')