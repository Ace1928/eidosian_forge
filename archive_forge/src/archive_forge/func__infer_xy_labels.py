from __future__ import annotations
import itertools
import textwrap
import warnings
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from datetime import date, datetime
from inspect import getfullargspec
from typing import TYPE_CHECKING, Any, Callable, Literal, overload
import numpy as np
import pandas as pd
from xarray.core.indexes import PandasMultiIndex
from xarray.core.options import OPTIONS
from xarray.core.utils import is_scalar, module_available
from xarray.namedarray.pycompat import DuckArrayModule
def _infer_xy_labels(darray: DataArray | Dataset, x: Hashable | None, y: Hashable | None, imshow: bool=False, rgb: Hashable | None=None) -> tuple[Hashable, Hashable]:
    """
    Determine x and y labels. For use in _plot2d

    darray must be a 2 dimensional data array, or 3d for imshow only.
    """
    if x is not None and x == y:
        raise ValueError('x and y cannot be equal.')
    if imshow and darray.ndim == 3:
        return _infer_xy_labels_3d(darray, x, y, rgb)
    if x is None and y is None:
        if darray.ndim != 2:
            raise ValueError('DataArray must be 2d')
        y, x = darray.dims
    elif x is None:
        _assert_valid_xy(darray, y, 'y')
        x = darray.dims[0] if y == darray.dims[1] else darray.dims[1]
    elif y is None:
        _assert_valid_xy(darray, x, 'x')
        y = darray.dims[0] if x == darray.dims[1] else darray.dims[1]
    else:
        _assert_valid_xy(darray, x, 'x')
        _assert_valid_xy(darray, y, 'y')
        if darray._indexes.get(x, 1) is darray._indexes.get(y, 2):
            if isinstance(darray._indexes[x], PandasMultiIndex):
                raise ValueError('x and y cannot be levels of the same MultiIndex')
    return (x, y)