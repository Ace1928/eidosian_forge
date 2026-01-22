import numpy
import cupy
from cupy import cublas
from cupyx import cusparse
from cupy._core import _dtype
from cupy.cuda import device
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupy_backends.cuda.libs import cublas as _cublas
from cupyx.scipy.sparse import _csr
from cupyx.scipy.sparse.linalg import _interface
Make a linear system Ax = b

    Args:
        A (cupy.ndarray or cupyx.scipy.sparse.spmatrix or
            cupyx.scipy.sparse.LinearOperator): sparse or dense matrix.
        M (cupy.ndarray or cupyx.scipy.sparse.spmatrix or
            cupyx.scipy.sparse.LinearOperator): preconditioner.
        x0 (cupy.ndarray): initial guess to iterative method.
        b (cupy.ndarray): right hand side.

    Returns:
        tuple:
            It returns (A, M, x, b).
            A (LinaerOperator): matrix of linear system
            M (LinearOperator): preconditioner
            x (cupy.ndarray): initial guess
            b (cupy.ndarray): right hand side.
    