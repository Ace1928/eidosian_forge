import warnings
import numpy
import cupy
import cupy.linalg as linalg
from cupyx.scipy.sparse import linalg as splinalg
def _eigh(A, B=None):
    """
    Helper function for converting a generalized eigenvalue problem
    A(X) = lambda(B(X)) to standard eigen value problem using cholesky
    transformation
    """
    if B is None:
        vals, vecs = linalg.eigh(A)
        return (vals, vecs)
    R = _cholesky(B)
    RTi = linalg.inv(R)
    Ri = linalg.inv(R.T)
    F = cupy.matmul(RTi, cupy.matmul(A, Ri))
    vals, vecs = linalg.eigh(F)
    eigVec = cupy.matmul(Ri, vecs)
    return (vals, eigVec)