from __future__ import annotations
from decimal import Decimal
from functools import partial
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import lib
import pandas._libs.missing as libmissing
from pandas._libs.tslibs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_list_like
def _isna_array(values: ArrayLike, inf_as_na: bool=False):
    """
    Return an array indicating which values of the input array are NaN / NA.

    Parameters
    ----------
    obj: ndarray or ExtensionArray
        The input array whose elements are to be checked.
    inf_as_na: bool
        Whether or not to treat infinite values as NA.

    Returns
    -------
    array-like
        Array of boolean values denoting the NA status of each element.
    """
    dtype = values.dtype
    if not isinstance(values, np.ndarray):
        if inf_as_na and isinstance(dtype, CategoricalDtype):
            result = libmissing.isnaobj(values.to_numpy(), inf_as_na=inf_as_na)
        else:
            result = values.isna()
    elif isinstance(values, np.rec.recarray):
        result = _isna_recarray_dtype(values, inf_as_na=inf_as_na)
    elif is_string_or_object_np_dtype(values.dtype):
        result = _isna_string_dtype(values, inf_as_na=inf_as_na)
    elif dtype.kind in 'mM':
        result = values.view('i8') == iNaT
    elif inf_as_na:
        result = ~np.isfinite(values)
    else:
        result = np.isnan(values)
    return result