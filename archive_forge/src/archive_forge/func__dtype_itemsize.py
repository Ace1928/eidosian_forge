import warnings
from warnings import warn
import numpy as np
def _dtype_itemsize(itemsize, *dtypes):
    """Return first of `dtypes` with itemsize greater than `itemsize`

    Parameters
    ----------
    itemsize: int
        The data type object element size.

    Other Parameters
    ----------------
    *dtypes:
        Any Object accepted by `np.dtype` to be converted to a data
        type object

    Returns
    -------
    dtype: data type object
        First of `dtypes` with itemsize greater than `itemsize`.

    """
    return next((dt for dt in dtypes if np.dtype(dt).itemsize >= itemsize))