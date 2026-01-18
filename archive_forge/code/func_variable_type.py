from __future__ import annotations
import warnings
from collections import UserString
from numbers import Number
from datetime import datetime
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING
def variable_type(vector: Series, boolean_type: Literal['numeric', 'categorical', 'boolean']='numeric', strict_boolean: bool=False) -> VarType:
    """
    Determine whether a vector contains numeric, categorical, or datetime data.

    This function differs from the pandas typing API in a few ways:

    - Python sequences or object-typed PyData objects are considered numeric if
      all of their entries are numeric.
    - String or mixed-type data are considered categorical even if not
      explicitly represented as a :class:`pandas.api.types.CategoricalDtype`.
    - There is some flexibility about how to treat binary / boolean data.

    Parameters
    ----------
    vector : :func:`pandas.Series`, :func:`numpy.ndarray`, or Python sequence
        Input data to test.
    boolean_type : 'numeric', 'categorical', or 'boolean'
        Type to use for vectors containing only 0s and 1s (and NAs).
    strict_boolean : bool
        If True, only consider data to be boolean when the dtype is bool or Boolean.

    Returns
    -------
    var_type : 'numeric', 'categorical', or 'datetime'
        Name identifying the type of data in the vector.
    """
    if isinstance(getattr(vector, 'dtype', None), pd.CategoricalDtype):
        return VarType('categorical')
    if pd.isna(vector).all():
        return VarType('numeric')
    vector = vector.dropna()
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=(FutureWarning, DeprecationWarning))
        if strict_boolean:
            if isinstance(vector.dtype, pd.core.dtypes.base.ExtensionDtype):
                boolean_dtypes = ['bool', 'boolean']
            else:
                boolean_dtypes = ['bool']
            boolean_vector = vector.dtype in boolean_dtypes
        else:
            try:
                boolean_vector = bool(np.isin(vector, [0, 1]).all())
            except TypeError:
                boolean_vector = False
        if boolean_vector:
            return VarType(boolean_type)
    if pd.api.types.is_numeric_dtype(vector):
        return VarType('numeric')
    if pd.api.types.is_datetime64_dtype(vector):
        return VarType('datetime')

    def all_numeric(x):
        for x_i in x:
            if not isinstance(x_i, Number):
                return False
        return True
    if all_numeric(vector):
        return VarType('numeric')

    def all_datetime(x):
        for x_i in x:
            if not isinstance(x_i, (datetime, np.datetime64)):
                return False
        return True
    if all_datetime(vector):
        return VarType('datetime')
    return VarType('categorical')