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
def _ensure_plottable(*args) -> None:
    """
    Raise exception if there is anything in args that can't be plotted on an
    axis by matplotlib.
    """
    numpy_types: tuple[type[object], ...] = (np.floating, np.integer, np.timedelta64, np.datetime64, np.bool_, np.str_)
    other_types: tuple[type[object], ...] = (datetime, date)
    cftime_datetime_types: tuple[type[object], ...] = () if cftime is None else (cftime.datetime,)
    other_types += cftime_datetime_types
    for x in args:
        if not (_valid_numpy_subdtype(np.asarray(x), numpy_types) or _valid_other_type(np.asarray(x), other_types)):
            raise TypeError(f'Plotting requires coordinates to be numeric, boolean, or dates of type numpy.datetime64, datetime.datetime, cftime.datetime or pandas.Interval. Received data of type {np.asarray(x).dtype} instead.')
        if _valid_other_type(np.asarray(x), cftime_datetime_types):
            if nc_time_axis_available:
                import nc_time_axis
            else:
                raise ImportError('Plotting of arrays of cftime.datetime objects or arrays indexed by cftime.datetime objects requires the optional `nc-time-axis` (v1.2.0 or later) package.')