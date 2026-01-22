import numpy as np
import scipy.sparse as sps
from ._numdiff import approx_derivative, group_columns
from ._hessian_update_strategy import HessianUpdateStrategy
from scipy.sparse.linalg import LinearOperator
from scipy._lib._array_api import atleast_nd, array_namespace
class LinearVectorFunction:
    """Linear vector function and its derivatives.

    Defines a linear function F = A x, where x is N-D vector and
    A is m-by-n matrix. The Jacobian is constant and equals to A. The Hessian
    is identically zero and it is returned as a csr matrix.
    """

    def __init__(self, A, x0, sparse_jacobian):
        if sparse_jacobian or (sparse_jacobian is None and sps.issparse(A)):
            self.J = sps.csr_matrix(A)
            self.sparse_jacobian = True
        elif sps.issparse(A):
            self.J = A.toarray()
            self.sparse_jacobian = False
        else:
            self.J = np.atleast_2d(np.asarray(A))
            self.sparse_jacobian = False
        self.m, self.n = self.J.shape
        self.xp = xp = array_namespace(x0)
        _x = atleast_nd(x0, ndim=1, xp=xp)
        _dtype = xp.float64
        if xp.isdtype(_x.dtype, 'real floating'):
            _dtype = _x.dtype
        self.x = xp.astype(_x, _dtype)
        self.x_dtype = _dtype
        self.f = self.J.dot(self.x)
        self.f_updated = True
        self.v = np.zeros(self.m, dtype=float)
        self.H = sps.csr_matrix((self.n, self.n))

    def _update_x(self, x):
        if not np.array_equal(x, self.x):
            _x = atleast_nd(x, ndim=1, xp=self.xp)
            self.x = self.xp.astype(_x, self.x_dtype)
            self.f_updated = False

    def fun(self, x):
        self._update_x(x)
        if not self.f_updated:
            self.f = self.J.dot(x)
            self.f_updated = True
        return self.f

    def jac(self, x):
        self._update_x(x)
        return self.J

    def hess(self, x, v):
        self._update_x(x)
        self.v = v
        return self.H