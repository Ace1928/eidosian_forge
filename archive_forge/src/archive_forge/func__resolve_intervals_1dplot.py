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
def _resolve_intervals_1dplot(xval: np.ndarray, yval: np.ndarray, kwargs: dict) -> tuple[np.ndarray, np.ndarray, str, str, dict]:
    """
    Helper function to replace the values of x and/or y coordinate arrays
    containing pd.Interval with their mid-points or - for step plots - double
    points which double the length.
    """
    x_suffix = ''
    y_suffix = ''
    if kwargs.get('drawstyle', '').startswith('steps-'):
        remove_drawstyle = False
        x_is_interval = _valid_other_type(xval, pd.Interval)
        y_is_interval = _valid_other_type(yval, pd.Interval)
        if x_is_interval and y_is_interval:
            raise TypeError("Can't step plot intervals against intervals.")
        elif x_is_interval:
            xval, yval = _interval_to_double_bound_points(xval, yval)
            remove_drawstyle = True
        elif y_is_interval:
            yval, xval = _interval_to_double_bound_points(yval, xval)
            remove_drawstyle = True
        if remove_drawstyle:
            del kwargs['drawstyle']
    else:
        if _valid_other_type(xval, pd.Interval):
            xval = _interval_to_mid_points(xval)
            x_suffix = '_center'
        if _valid_other_type(yval, pd.Interval):
            yval = _interval_to_mid_points(yval)
            y_suffix = '_center'
    return (xval, yval, x_suffix, y_suffix, kwargs)