import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import kron, eye, dia_array
class MikotaM(LinearOperator):
    """
    Construct a mass matrix in various formats of Mikota pair.

    The mass matrix `M` is square real diagonal
    positive definite with entries that are reciprocal to integers.

    Parameters
    ----------
    shape : tuple of int
        The shape of the matrix.
    dtype : dtype
        Numerical type of the array. Default is ``np.float64``.

    Methods
    -------
    toarray()
        Construct a dense array from Mikota data
    tosparse()
        Construct a sparse array from Mikota data
    tobanded()
        The format for banded symmetric matrices,
        i.e., (1, n) ndarray with the main diagonal.
    """

    def __init__(self, shape, dtype=np.float64):
        self.shape = shape
        self.dtype = dtype
        super().__init__(dtype, shape)

    def _diag(self):
        return (1.0 / np.arange(1, self.shape[0] + 1)).astype(self.dtype)

    def tobanded(self):
        return self._diag()

    def tosparse(self):
        from scipy.sparse import diags
        return diags([self._diag()], [0], shape=self.shape, dtype=self.dtype)

    def toarray(self):
        return np.diag(self._diag()).astype(self.dtype)

    def _matvec(self, x):
        """
        Construct matrix-free callable banded-matrix-vector multiplication by
        the Mikota mass matrix without constructing or storing the matrix itself
        using the knowledge of its entries and the diagonal format.
        """
        x = x.reshape(self.shape[0], -1)
        return self._diag()[:, np.newaxis] * x

    def _matmat(self, x):
        """
        Construct matrix-free callable matrix-matrix multiplication by
        the Mikota mass matrix without constructing or storing the matrix itself
        by reusing the ``_matvec(x)`` that supports both 1D and 2D arrays ``x``.
        """
        return self._matvec(x)

    def _adjoint(self):
        return self

    def _transpose(self):
        return self