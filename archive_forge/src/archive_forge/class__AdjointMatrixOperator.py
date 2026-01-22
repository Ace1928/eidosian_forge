import warnings
import cupy
from cupyx.scipy import sparse
from cupyx.scipy.sparse import _util
class _AdjointMatrixOperator(MatrixLinearOperator):

    def __init__(self, adjoint):
        self.A = adjoint.A.T.conj()
        self.__adjoint = adjoint
        self.args = (adjoint,)
        self.shape = (adjoint.shape[1], adjoint.shape[0])

    @property
    def dtype(self):
        return self.__adjoint.dtype

    def _adjoint(self):
        return self.__adjoint