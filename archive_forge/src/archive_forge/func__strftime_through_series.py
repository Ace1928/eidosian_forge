from __future__ import annotations
import warnings
from typing import TYPE_CHECKING, Generic
import numpy as np
import pandas as pd
from xarray.coding.times import infer_calendar_name
from xarray.core import duck_array_ops
from xarray.core.common import (
from xarray.core.types import T_DataArray
from xarray.core.variable import IndexVariable
from xarray.namedarray.utils import is_duck_dask_array
def _strftime_through_series(values, date_format: str):
    """Coerce an array of datetime-like values to a pandas Series and
    apply string formatting
    """
    values_as_series = pd.Series(duck_array_ops.ravel(values), copy=False)
    strs = values_as_series.dt.strftime(date_format)
    return strs.values.reshape(values.shape)