from decorator import decorator
from scipy import sparse
import importlib
import numbers
import numpy as np
import pandas as pd
import re
import warnings
def to_array_or_spmatrix(x):
    """Convert an array-like to a np.ndarray or scipy.sparse.spmatrix.

    Parameters
    ----------
    x : array-like
        Array-like to be converted
    Returns
    -------
    x : np.ndarray or scipy.sparse.spmatrix
    """
    if is_SparseDataFrame(x):
        x = x.to_coo()
    elif is_sparse_dataframe(x) or is_sparse_series(x):
        x = x.sparse.to_coo()
    elif isinstance(x, (sparse.spmatrix, np.ndarray, numbers.Number)) and (not isinstance(x, np.matrix)):
        pass
    elif isinstance(x, list):
        x_out = []
        for xi in x:
            try:
                xi = to_array_or_spmatrix(xi)
            except TypeError:
                pass
            x_out.append(xi)
        x = np.array(x_out, dtype=_check_numpy_dtype(x_out))
    else:
        x = toarray(x)
    return x