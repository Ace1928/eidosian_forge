import json
from array import array
from enum import Enum, auto
from typing import Any
def _copytobuffer(xxx: Any, inplace: bool=False) -> tuple[Any, DataType]:
    """
    Prepares data for PROJ C-API:
    - Makes a copy because PROJ modifies buffer in place
    - Make sure dtype is double as that is what PROJ expects
    - Makes sure object supports Python Buffer API

    If the data is a numpy array, it ensures the data is in C order.

    Parameters
    ----------
    xxx: Any
        A scalar, list, tuple, numpy.array,
        pandas.Series, xaray.DataArray, or dask.array.Array.
    inplace: bool, default=False
        If True, will return the array without a copy if it
        meets the requirements of the Python Buffer API & PROJ C-API.

    Returns
    -------
    tuple[Any, DataType]
        The copy of the data prepared for the PROJ API & Python Buffer API.
    """
    if not hasattr(xxx, 'hardmask') and hasattr(xxx, '__array__') and callable(xxx.__array__):
        xxx = xxx.__array__()
    if hasattr(xxx, 'shape'):
        if xxx.shape == ():
            return _copytobuffer_return_scalar(xxx)
        return (xxx.astype('d', order='C', copy=not inplace), DataType.ARRAY)
    data_type = DataType.ARRAY
    if isinstance(xxx, array):
        if not inplace or xxx.typecode != 'd':
            xxx = array('d', xxx)
    elif isinstance(xxx, list):
        xxx = array('d', xxx)
        data_type = DataType.LIST
    elif isinstance(xxx, tuple):
        xxx = array('d', xxx)
        data_type = DataType.TUPLE
    else:
        return _copytobuffer_return_scalar(xxx)
    return (xxx, data_type)