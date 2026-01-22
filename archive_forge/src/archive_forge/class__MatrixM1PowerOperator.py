import warnings
import numpy as np
from scipy.linalg._matfuncs_sqrtm import SqrtmError, _sqrtm_triu
from scipy.linalg._decomp_schur import schur, rsf2csf
from scipy.linalg._matfuncs import funm
from scipy.linalg import svdvals, solve_triangular
from scipy.sparse.linalg._interface import LinearOperator
from scipy.sparse.linalg import onenormest
import scipy.special
class _MatrixM1PowerOperator(LinearOperator):
    """
    A representation of the linear operator (A - I)^p.
    """

    def __init__(self, A, p):
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError('expected A to be like a square matrix')
        if p < 0 or p != int(p):
            raise ValueError('expected p to be a non-negative integer')
        self._A = A
        self._p = p
        self.ndim = A.ndim
        self.shape = A.shape

    def _matvec(self, x):
        for i in range(self._p):
            x = self._A.dot(x) - x
        return x

    def _rmatvec(self, x):
        for i in range(self._p):
            x = x.dot(self._A) - x
        return x

    def _matmat(self, X):
        for i in range(self._p):
            X = self._A.dot(X) - X
        return X

    def _adjoint(self):
        return _MatrixM1PowerOperator(self._A.T, self._p)