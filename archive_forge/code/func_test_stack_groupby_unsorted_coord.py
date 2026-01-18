from __future__ import annotations
import datetime
import operator
import warnings
from unittest import mock
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import xarray as xr
from xarray import DataArray, Dataset, Variable
from xarray.core.groupby import _consolidate_slices
from xarray.core.types import InterpOptions
from xarray.tests import (
def test_stack_groupby_unsorted_coord(self) -> None:
    data = [[0, 1], [2, 3]]
    data_flat = [0, 1, 2, 3]
    dims = ['x', 'y']
    y_vals = [2, 3]
    arr = xr.DataArray(data, dims=dims, coords={'y': y_vals})
    actual1 = arr.stack(z=dims).groupby('z').first()
    midx1 = pd.MultiIndex.from_product([[0, 1], [2, 3]], names=dims)
    expected1 = xr.DataArray(data_flat, dims=['z'], coords={'z': midx1})
    assert_equal(actual1, expected1)
    arr = xr.DataArray(data, dims=dims, coords={'y': y_vals[::-1]})
    actual2 = arr.stack(z=dims).groupby('z').first()
    midx2 = pd.MultiIndex.from_product([[0, 1], [3, 2]], names=dims)
    expected2 = xr.DataArray(data_flat, dims=['z'], coords={'z': midx2})
    assert_equal(actual2, expected2)