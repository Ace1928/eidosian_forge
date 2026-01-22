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
def _strftime_through_cftimeindex(values, date_format: str):
    """Coerce an array of cftime-like values to a CFTimeIndex
    and access requested datetime component
    """
    from xarray.coding.cftimeindex import CFTimeIndex
    values_as_cftimeindex = CFTimeIndex(duck_array_ops.ravel(values))
    field_values = values_as_cftimeindex.strftime(date_format)
    return field_values.values.reshape(values.shape)