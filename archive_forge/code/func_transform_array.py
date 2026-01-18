from __future__ import annotations
import logging # isort:skip
import datetime as dt
import uuid
from functools import lru_cache
from threading import Lock
from typing import TYPE_CHECKING, Any
import numpy as np
from ..core.types import ID
from ..settings import settings
from .strings import format_docstring
def transform_array(array: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """ Transform a ndarray into a serializable ndarray.

    Converts un-serializable dtypes and returns JSON serializable
    format

    Args:
        array (np.ndarray) : a NumPy array to be transformed

    Returns:
        ndarray

    """
    array = convert_datetime_array(array)

    def _cast_if_can(array: npt.NDArray[Any], dtype: type[Any]) -> npt.NDArray[Any]:
        info = np.iinfo(dtype)
        if np.any((array < info.min) | (info.max < array)):
            return array
        else:
            return array.astype(dtype, casting='unsafe')
    if array.dtype == np.dtype(np.int64):
        array = _cast_if_can(array, np.int32)
    elif array.dtype == np.dtype(np.uint64):
        array = _cast_if_can(array, np.uint32)
    if isinstance(array, np.ma.MaskedArray):
        array = array.filled(np.nan)
    if not array.flags['C_CONTIGUOUS']:
        array = np.ascontiguousarray(array)
    return array