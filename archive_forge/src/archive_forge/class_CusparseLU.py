import numpy
import cupy
import cupyx.cusolver
from cupy import cublas
from cupyx import cusparse
from cupy.cuda import cusolver
from cupy.cuda import device
from cupy.cuda import runtime
from cupy.linalg import _util
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupyx.scipy import sparse
from cupyx.scipy.sparse.linalg import _interface
from cupyx.scipy.sparse.linalg._iterative import _make_system
import warnings
class CusparseLU(SuperLU):

    def __init__(self, a):
        """Incomplete LU factorization of a sparse matrix.

        Args:
            a (cupyx.scipy.sparse.csr_matrix): Incomplete LU factorization of a
                sparse matrix, computed by `cusparse.csrilu02`.
        """
        if not scipy_available:
            raise RuntimeError('scipy is not available')
        if not sparse.isspmatrix_csr(a):
            raise TypeError('a must be cupyx.scipy.sparse.csr_matrix')
        self.shape = a.shape
        self.nnz = a.nnz
        self.perm_r = None
        self.perm_c = None
        a = a.get()
        al = scipy.sparse.tril(a)
        al.setdiag(1.0)
        au = scipy.sparse.triu(a)
        self.L = sparse.csr_matrix(al.tocsr())
        self.U = sparse.csr_matrix(au.tocsr())