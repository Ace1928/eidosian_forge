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
@requires_iris
def test_to_and_from_iris(self) -> None:
    import cf_units
    import iris
    coord_dict: dict[Hashable, Any] = {}
    coord_dict['distance'] = ('distance', [-2, 2], {'units': 'meters'})
    coord_dict['time'] = ('time', pd.date_range('2000-01-01', periods=3))
    coord_dict['height'] = 10
    coord_dict['distance2'] = ('distance', [0, 1], {'foo': 'bar'})
    coord_dict['time2'] = (('distance', 'time'), [[0, 1, 2], [2, 3, 4]])
    original = DataArray(np.arange(6, dtype='float').reshape(2, 3), coord_dict, name='Temperature', attrs={'baz': 123, 'units': 'Kelvin', 'standard_name': 'fire_temperature', 'long_name': 'Fire Temperature'}, dims=('distance', 'time'))
    original.data[0, 2] = np.nan
    original.attrs['cell_methods'] = 'height: mean (comment: A cell method)'
    actual = original.to_iris()
    assert_array_equal(actual.data, original.data)
    assert actual.var_name == original.name
    assert tuple((d.var_name for d in actual.dim_coords)) == original.dims
    assert actual.cell_methods == (iris.coords.CellMethod(method='mean', coords=('height',), intervals=(), comments=('A cell method',)),)
    for coord, orginal_key in zip(actual.coords(), original.coords):
        original_coord = original.coords[orginal_key]
        assert coord.var_name == original_coord.name
        assert_array_equal(coord.points, CFDatetimeCoder().encode(original_coord.variable).values)
        assert actual.coord_dims(coord) == original.get_axis_num(original.coords[coord.var_name].dims)
    assert actual.coord('distance2').attributes['foo'] == original.coords['distance2'].attrs['foo']
    assert actual.coord('distance').units == cf_units.Unit(original.coords['distance'].units)
    assert actual.attributes['baz'] == original.attrs['baz']
    assert actual.standard_name == original.attrs['standard_name']
    roundtripped = DataArray.from_iris(actual)
    assert_identical(original, roundtripped)
    actual.remove_coord('time')
    auto_time_dimension = DataArray.from_iris(actual)
    assert auto_time_dimension.dims == ('distance', 'dim_1')