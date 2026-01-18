import numpy as np
import scipy.linalg
def vec(M):
    """
    Stack columns of M to construct a single vector.

    This is somewhat standard notation in linear algebra.

    Parameters
    ----------
    M : 2-D array_like
        Input matrix

    Returns
    -------
    v : 1-D ndarray
        Output vector

    """
    return M.T.ravel()