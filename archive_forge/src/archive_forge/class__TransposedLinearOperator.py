import warnings
import cupy
from cupyx.scipy import sparse
from cupyx.scipy.sparse import _util
class _TransposedLinearOperator(LinearOperator):
    """Transposition of arbitrary Linear Operator"""

    def __init__(self, A):
        shape = (A.shape[1], A.shape[0])
        super(_TransposedLinearOperator, self).__init__(dtype=A.dtype, shape=shape)
        self.A = A
        self.args = (A,)

    def _matvec(self, x):
        return cupy.conj(self.A._rmatvec(cupy.conj(x)))

    def _rmatvec(self, x):
        return cupy.conj(self.A._matvec(cupy.conj(x)))

    def _matmat(self, x):
        return cupy.conj(self.A._rmatmat(cupy.conj(x)))

    def _rmatmat(self, x):
        return cupy.conj(self.A._matmat(cupy.conj(x)))