import numpy as np
import scipy.sparse
from scipy.ndimage import median_filter
import sklearn.decomposition
from . import core
from ._cache import cache
from . import segment
from . import util
from .util.exceptions import ParameterError
from typing import Any, Callable, List, Optional, Tuple, Union
from ._typing import _IntLike_co, _FloatLike_co
def __nn_filter_helper(R_data, R_indices, R_ptr, S: np.ndarray, aggregate: Callable) -> np.ndarray:
    """Nearest-neighbor filter helper function.

    This is an internal function, not for use outside of the decompose module.

    It applies the nearest-neighbor filter to S, assuming that the first index
    corresponds to observations.

    Parameters
    ----------
    R_data, R_indices, R_ptr : np.ndarrays
        The ``data``, ``indices``, and ``indptr`` of a scipy.sparse matrix
    S : np.ndarray
        The observation data to filter
    aggregate : callable
        The aggregation operator

    Returns
    -------
    S_out : np.ndarray like S
        The filtered data array
    """
    s_out = np.empty_like(S)
    for i in range(len(R_ptr) - 1):
        targets = R_indices[R_ptr[i]:R_ptr[i + 1]]
        if not len(targets):
            s_out[i] = S[i]
            continue
        neighbors = np.take(S, targets, axis=0)
        if aggregate is np.average:
            weights = R_data[R_ptr[i]:R_ptr[i + 1]]
            s_out[i] = aggregate(neighbors, axis=0, weights=weights)
        else:
            s_out[i] = aggregate(neighbors, axis=0)
    return s_out