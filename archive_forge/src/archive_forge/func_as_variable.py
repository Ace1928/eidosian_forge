from __future__ import annotations
import copy
import itertools
import math
import numbers
import warnings
from collections.abc import Hashable, Mapping, Sequence
from datetime import timedelta
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Literal, NoReturn, cast
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
import xarray as xr  # only for Dataset and DataArray
from xarray.core import common, dtypes, duck_array_ops, indexing, nputils, ops, utils
from xarray.core.arithmetic import VariableArithmetic
from xarray.core.common import AbstractArray
from xarray.core.indexing import (
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.utils import (
from xarray.namedarray.core import NamedArray, _raise_if_any_duplicate_dimensions
from xarray.namedarray.pycompat import integer_types, is_0d_dask_array, to_duck_array
def as_variable(obj: T_DuckArray | Any, name=None, auto_convert: bool=True) -> Variable | IndexVariable:
    """Convert an object into a Variable.

    Parameters
    ----------
    obj : object
        Object to convert into a Variable.

        - If the object is already a Variable, return a shallow copy.
        - Otherwise, if the object has 'dims' and 'data' attributes, convert
          it into a new Variable.
        - If all else fails, attempt to convert the object into a Variable by
          unpacking it into the arguments for creating a new Variable.
    name : str, optional
        If provided:

        - `obj` can be a 1D array, which is assumed to label coordinate values
          along a dimension of this given name.
        - Variables with name matching one of their dimensions are converted
          into `IndexVariable` objects.
    auto_convert : bool, optional
        For internal use only! If True, convert a "dimension" variable into
        an IndexVariable object (deprecated).

    Returns
    -------
    var : Variable
        The newly created variable.

    """
    from xarray.core.dataarray import DataArray
    if isinstance(obj, DataArray):
        obj = obj.variable
    if isinstance(obj, Variable):
        obj = obj.copy(deep=False)
    elif isinstance(obj, tuple):
        if isinstance(obj[1], DataArray):
            raise TypeError(f'Variable {name!r}: Using a DataArray object to construct a variable is ambiguous, please extract the data using the .data property.')
        try:
            obj = Variable(*obj)
        except (TypeError, ValueError) as error:
            raise error.__class__(f'Variable {name!r}: Could not convert tuple of form (dims, data[, attrs, encoding]): {obj} to Variable.')
    elif utils.is_scalar(obj):
        obj = Variable([], obj)
    elif isinstance(obj, (pd.Index, IndexVariable)) and obj.name is not None:
        obj = Variable(obj.name, obj)
    elif isinstance(obj, (set, dict)):
        raise TypeError(f'variable {name!r} has invalid type {type(obj)!r}')
    elif name is not None:
        data: T_DuckArray = as_compatible_data(obj)
        if data.ndim != 1:
            raise MissingDimensionsError(f'cannot set variable {name!r} with {data.ndim!r}-dimensional data without explicit dimension names. Pass a tuple of (dims, data) instead.')
        obj = Variable(name, data, fastpath=True)
    else:
        raise TypeError(f'Variable {name!r}: unable to convert object into a variable without an explicit list of dimensions: {obj!r}')
    if auto_convert:
        if name is not None and name in obj.dims and (obj.ndim == 1):
            emit_user_level_warning(f'variable {name!r} with name matching its dimension will not be automatically converted into an `IndexVariable` object in the future.', FutureWarning)
            obj = obj.to_index_variable()
    return obj