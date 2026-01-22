import numpy as np
import pandas as pd
import scipy.linalg
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.validation import array_like
def _ensure_2d(x, ndarray=False):
    """

    Parameters
    ----------
    x : ndarray, Series, DataFrame or None
        Input to verify dimensions, and to transform as necesary
    ndarray : bool
        Flag indicating whether to always return a NumPy array. Setting False
        will return an pandas DataFrame when the input is a Series or a
        DataFrame.

    Returns
    -------
    out : ndarray, DataFrame or None
        array or DataFrame with 2 dimensiona.  One dimensional arrays are
        returned as nobs by 1. None is returned if x is None.
    names : list of str or None
        list containing variables names when the input is a pandas datatype.
        Returns None if the input is an ndarray.

    Notes
    -----
    Accepts None for simplicity
    """
    if x is None:
        return x
    is_pandas = _is_using_pandas(x, None)
    if x.ndim == 2:
        if is_pandas:
            return (x, x.columns)
        else:
            return (x, None)
    elif x.ndim > 2:
        raise ValueError('x mst be 1 or 2-dimensional.')
    name = x.name if is_pandas else None
    if ndarray:
        return (np.asarray(x)[:, None], name)
    else:
        return (pd.DataFrame(x), name)