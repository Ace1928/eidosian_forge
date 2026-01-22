from __future__ import annotations
import inspect
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.tslibs.timedeltas import array_to_timedelta64
from pandas.errors import IntCastingNaNError
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
def _astype_nansafe(arr: np.ndarray, dtype: DtypeObj, copy: bool=True, skipna: bool=False) -> ArrayLike:
    """
    Cast the elements of an array to a given dtype a nan-safe manner.

    Parameters
    ----------
    arr : ndarray
    dtype : np.dtype or ExtensionDtype
    copy : bool, default True
        If False, a view will be attempted but may fail, if
        e.g. the item sizes don't align.
    skipna: bool, default False
        Whether or not we should skip NaN when casting as a string-type.

    Raises
    ------
    ValueError
        The dtype was a datetime64/timedelta64 dtype, but it had no unit.
    """
    if isinstance(dtype, ExtensionDtype):
        return dtype.construct_array_type()._from_sequence(arr, dtype=dtype, copy=copy)
    elif not isinstance(dtype, np.dtype):
        raise ValueError('dtype must be np.dtype or ExtensionDtype')
    if arr.dtype.kind in 'mM':
        from pandas.core.construction import ensure_wrapped_if_datetimelike
        arr = ensure_wrapped_if_datetimelike(arr)
        res = arr.astype(dtype, copy=copy)
        return np.asarray(res)
    if issubclass(dtype.type, str):
        shape = arr.shape
        if arr.ndim > 1:
            arr = arr.ravel()
        return lib.ensure_string_array(arr, skipna=skipna, convert_na_value=False).reshape(shape)
    elif np.issubdtype(arr.dtype, np.floating) and dtype.kind in 'iu':
        return _astype_float_to_int_nansafe(arr, dtype, copy)
    elif arr.dtype == object:
        if lib.is_np_dtype(dtype, 'M'):
            from pandas.core.arrays import DatetimeArray
            dta = DatetimeArray._from_sequence(arr, dtype=dtype)
            return dta._ndarray
        elif lib.is_np_dtype(dtype, 'm'):
            from pandas.core.construction import ensure_wrapped_if_datetimelike
            tdvals = array_to_timedelta64(arr).view('m8[ns]')
            tda = ensure_wrapped_if_datetimelike(tdvals)
            return tda.astype(dtype, copy=False)._ndarray
    if dtype.name in ('datetime64', 'timedelta64'):
        msg = f"The '{dtype.name}' dtype has no unit. Please pass in '{dtype.name}[ns]' instead."
        raise ValueError(msg)
    if copy or arr.dtype == object or dtype == object:
        return arr.astype(dtype, copy=True)
    return arr.astype(dtype, copy=copy)