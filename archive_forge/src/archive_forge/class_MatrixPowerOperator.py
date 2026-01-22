import numpy as np
from scipy.linalg._basic import solve, solve_triangular
from scipy.sparse._base import issparse
from scipy.sparse.linalg import spsolve
from scipy.sparse._sputils import is_pydata_spmatrix, isintlike
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.linalg._interface import LinearOperator
from scipy.sparse._construct import eye
from ._expm_multiply import _ident_like, _exact_1_norm as _onenorm
class MatrixPowerOperator(LinearOperator):

    def __init__(self, A, p, structure=None):
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError('expected A to be like a square matrix')
        if p < 0:
            raise ValueError('expected p to be a non-negative integer')
        self._A = A
        self._p = p
        self._structure = structure
        self.dtype = A.dtype
        self.ndim = A.ndim
        self.shape = A.shape

    def _matvec(self, x):
        for i in range(self._p):
            x = self._A.dot(x)
        return x

    def _rmatvec(self, x):
        A_T = self._A.T
        x = x.ravel()
        for i in range(self._p):
            x = A_T.dot(x)
        return x

    def _matmat(self, X):
        for i in range(self._p):
            X = _smart_matrix_product(self._A, X, structure=self._structure)
        return X

    @property
    def T(self):
        return MatrixPowerOperator(self._A.T, self._p)