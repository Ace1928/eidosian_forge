import numpy as np
from numpy.linalg import lstsq
def _lowess_wt_standardize(weights, new_entries, x_copy_i, width):
    """
    The initial phase of creating the weights.
    Subtract the current x_i and divide by the width.

    Parameters
    ----------
    weights : ndarray
        The memory where (new_entries - x_copy_i)/width will be placed
    new_entries : ndarray
        The x-values of the k closest points to x[i]
    x_copy_i : float
        x[i], the i'th point in the (sorted) x values
    width : float
        The maximum distance between x[i] and any point in new_entries

    Returns
    -------
    Nothing. The modifications are made to weight in place.
    """
    weights[:] = new_entries
    weights -= x_copy_i
    weights /= width