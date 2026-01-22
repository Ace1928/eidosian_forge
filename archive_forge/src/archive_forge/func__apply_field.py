import io
import os
import numpy as np
import scipy.sparse
from scipy.io import _mmio
def _apply_field(data, field, no_pattern=False):
    """
    Ensure that ``data.dtype`` is compatible with the specified MatrixMarket field type.

    Parameters
    ----------
    data : ndarray
        Input array.

    field : str
        Matrix Market field, such as 'real', 'complex', 'integer', 'pattern'.

    no_pattern : bool, optional
        Whether an empty array may be returned for a 'pattern' field.

    Returns
    -------
    data : ndarray
        Input data if no conversion necessary, or a converted version
    """
    if field is None:
        return data
    if field == 'pattern':
        if no_pattern:
            return data
        else:
            return np.zeros(0)
    dtype = _field_to_dtype.get(field, None)
    if dtype is None:
        raise ValueError('Invalid field.')
    return np.asarray(data, dtype=dtype)