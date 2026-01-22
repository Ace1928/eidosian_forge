import numpy as np
from scipy._lib._util import check_random_state
from scipy.sparse.linalg import aslinearoperator
from scipy.linalg import LinAlgError
from ._propack import _spropack  # type: ignore[attr-defined]
from ._propack import _dpropack  # type: ignore[attr-defined]
from ._propack import _cpropack  # type: ignore[attr-defined]
from ._propack import _zpropack  # type: ignore[attr-defined]
class _AProd:
    """
    Wrapper class for linear operator

    The call signature of the __call__ method matches the callback of
    the PROPACK routines.
    """

    def __init__(self, A):
        try:
            self.A = aslinearoperator(A)
        except TypeError:
            self.A = aslinearoperator(np.asarray(A))

    def __call__(self, transa, m, n, x, y, sparm, iparm):
        if transa == 'n':
            y[:] = self.A.matvec(x)
        else:
            y[:] = self.A.rmatvec(x)

    @property
    def shape(self):
        return self.A.shape

    @property
    def dtype(self):
        try:
            return self.A.dtype
        except AttributeError:
            return self.A.matvec(np.zeros(self.A.shape[1])).dtype