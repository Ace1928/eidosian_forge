from __future__ import annotations
import pickle
import re
import sys
import warnings
from collections.abc import Hashable
from copy import deepcopy
from textwrap import dedent
from typing import Any, Final, Literal, cast
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import (
from xarray.coding.times import CFDatetimeCoder
from xarray.core import dtypes
from xarray.core.common import full_like
from xarray.core.coordinates import Coordinates
from xarray.core.indexes import Index, PandasIndex, filter_indexes_from_coords
from xarray.core.types import QueryEngineOptions, QueryParserOptions
from xarray.core.utils import is_scalar
from xarray.testing import _assert_internal_invariants
from xarray.tests import (
def test_to_dataframe(self) -> None:
    arr_np = np.random.randn(3, 4)
    arr = DataArray(arr_np, [('B', [1, 2, 3]), ('A', list('cdef'))], name='foo')
    expected = arr.to_series()
    actual = arr.to_dataframe()['foo']
    assert_array_equal(expected.values, actual.values)
    assert_array_equal(expected.name, actual.name)
    assert_array_equal(expected.index.values, actual.index.values)
    actual = arr.to_dataframe(dim_order=['A', 'B'])['foo']
    assert_array_equal(arr_np.transpose().reshape(-1), actual.values)
    arr.coords['C'] = ('B', [-1, -2, -3])
    expected = arr.to_series().to_frame()
    expected['C'] = [-1] * 4 + [-2] * 4 + [-3] * 4
    expected = expected[['C', 'foo']]
    actual = arr.to_dataframe()
    assert_array_equal(expected.values, actual.values)
    assert_array_equal(expected.columns.values, actual.columns.values)
    assert_array_equal(expected.index.values, actual.index.values)
    with pytest.raises(ValueError, match='does not match the set of dimensions'):
        arr.to_dataframe(dim_order=['B', 'A', 'C'])
    with pytest.raises(ValueError, match='cannot convert a scalar'):
        arr.sel(A='c', B=2).to_dataframe()
    arr.name = None
    with pytest.raises(ValueError, match='unnamed'):
        arr.to_dataframe()