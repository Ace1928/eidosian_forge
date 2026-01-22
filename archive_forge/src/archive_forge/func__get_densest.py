import numpy as np
from scipy.linalg import svd
from scipy.linalg.interpolative import interp_decomp
import scipy
from scipy.linalg.blas import dtrsm
def _get_densest(A, eligibleRows):
    """
    Returns the index of the densest row of A. Ignores rows that are not
    eligible for consideration.

    Parameters
    ----------
    A : 2-D array
        An array representing a matrix
    eligibleRows : 1-D logical array
        Values indicate whether the corresponding row of A is eligible
        to be considered

    Returns
    -------
    i_densest : int
        Index of the densest row in A eligible for consideration

    """
    rowCounts = _row_count(A)
    return np.argmax(rowCounts * eligibleRows)