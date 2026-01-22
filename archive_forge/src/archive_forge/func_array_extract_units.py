from __future__ import annotations
import functools
import operator
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, duck_array_ops
from xarray.tests import (
from xarray.tests.test_plot import PlotTestCase
from xarray.tests.test_variable import _PAD_XR_NP_ARGS
def array_extract_units(obj):
    if isinstance(obj, (xr.Variable, xr.DataArray, xr.Dataset)):
        obj = obj.data
    try:
        return obj.units
    except AttributeError:
        return None