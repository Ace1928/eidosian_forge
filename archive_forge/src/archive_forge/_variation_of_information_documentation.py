import numpy as np
import scipy.sparse as sparse
from ._contingency_table import contingency_table
from .._shared.utils import check_shape_equality
Compute the inverse of the non-zero elements of arr, not changing 0.

    Parameters
    ----------
    arr : ndarray

    Returns
    -------
    arr_inv : ndarray
        Array containing the inverse of the non-zero elements of arr, and
        zero elsewhere.
    