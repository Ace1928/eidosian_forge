from __future__ import annotations
import pickle
import re
import sys
import warnings
from collections.abc import Hashable
from copy import copy, deepcopy
from io import StringIO
from textwrap import dedent
from typing import Any, Literal
import numpy as np
import pandas as pd
import pytest
from pandas.core.indexes.datetimes import DatetimeIndex
import xarray as xr
from xarray import (
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core import dtypes, indexing, utils
from xarray.core.common import duck_array_ops, full_like
from xarray.core.coordinates import Coordinates, DatasetCoordinates
from xarray.core.indexes import Index, PandasIndex
from xarray.core.utils import is_scalar
from xarray.namedarray.pycompat import array_type, integer_types
from xarray.testing import _assert_internal_invariants
from xarray.tests import (
def test_sel_dataarray_mindex(self) -> None:
    midx = pd.MultiIndex.from_product([list('abc'), [0, 1]], names=('one', 'two'))
    midx_coords = Coordinates.from_pandas_multiindex(midx, 'x')
    midx_coords['y'] = range(3)
    mds = xr.Dataset({'var': (('x', 'y'), np.random.rand(6, 3))}, coords=midx_coords)
    actual_isel = mds.isel(x=xr.DataArray(np.arange(3), dims='x'))
    actual_sel = mds.sel(x=DataArray(midx[:3], dims='x'))
    assert actual_isel['x'].dims == ('x',)
    assert actual_sel['x'].dims == ('x',)
    assert_identical(actual_isel, actual_sel)
    actual_isel = mds.isel(x=xr.DataArray(np.arange(3), dims='z'))
    actual_sel = mds.sel(x=Variable('z', midx[:3]))
    assert actual_isel['x'].dims == ('z',)
    assert actual_sel['x'].dims == ('z',)
    assert_identical(actual_isel, actual_sel)
    actual_isel = mds.isel(x=xr.DataArray(np.arange(3), dims='z', coords={'z': [0, 1, 2]}))
    actual_sel = mds.sel(x=xr.DataArray(midx[:3], dims='z', coords={'z': [0, 1, 2]}))
    assert actual_isel['x'].dims == ('z',)
    assert actual_sel['x'].dims == ('z',)
    assert_identical(actual_isel, actual_sel)
    with pytest.raises(ValueError, match='Vectorized selection is '):
        mds.sel(one=['a', 'b'])
    with pytest.raises(ValueError, match="Vectorized selection is not available along coordinate 'x' with a multi-index"):
        mds.sel(x=xr.DataArray([np.array(midx[:2]), np.array(midx[-2:])], dims=['a', 'b']))