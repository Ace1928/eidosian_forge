import numpy as np
import scipy.sparse as sparse
from ._contingency_table import contingency_table
from .._shared.utils import check_shape_equality
def _xlogx(x):
    """Compute x * log_2(x).

    We define 0 * log_2(0) = 0

    Parameters
    ----------
    x : ndarray or scipy.sparse.csc_matrix or csr_matrix
        The input array.

    Returns
    -------
    y : same type as x
        Result of x * log_2(x).
    """
    y = x.copy()
    if isinstance(y, sparse.csc_matrix) or isinstance(y, sparse.csr_matrix):
        z = y.data
    else:
        z = np.asarray(y)
    nz = z.nonzero()
    z[nz] *= np.log2(z[nz])
    return y