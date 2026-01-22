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
def _check_symmetric(op1, op2, vec, eps):
    r2 = op1 * op2
    s = cupy.inner(op2, op2)
    t = cupy.inner(vec, r2)
    z = abs(s - t)
    epsa = (s + eps) * eps ** (1.0 / 3.0)
    if z > epsa:
        return False
    return True