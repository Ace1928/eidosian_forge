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
class LuInv(LinearOperator):
    """
    LuInv:
       helper class to repeatedly solve M*x=b
       using an LU-decomposition of M
    """

    def __init__(self, M):
        self.M_lu = lu_factor(M)
        self.shape = M.shape
        self.dtype = M.dtype

    def _matvec(self, x):
        return lu_solve(self.M_lu, x)