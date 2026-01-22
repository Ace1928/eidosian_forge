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
class TestIrisConversion:

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

    @requires_iris
    @requires_dask
    def test_to_and_from_iris_dask(self) -> None:
        import cf_units
        import dask.array as da
        import iris
        coord_dict: dict[Hashable, Any] = {}
        coord_dict['distance'] = ('distance', [-2, 2], {'units': 'meters'})
        coord_dict['time'] = ('time', pd.date_range('2000-01-01', periods=3))
        coord_dict['height'] = 10
        coord_dict['distance2'] = ('distance', [0, 1], {'foo': 'bar'})
        coord_dict['time2'] = (('distance', 'time'), [[0, 1, 2], [2, 3, 4]])
        original = DataArray(da.from_array(np.arange(-1, 5, dtype='float').reshape(2, 3), 3), coord_dict, name='Temperature', attrs=dict(baz=123, units='Kelvin', standard_name='fire_temperature', long_name='Fire Temperature'), dims=('distance', 'time'))
        original.data = da.ma.masked_less(original.data, 0)
        original.attrs['cell_methods'] = 'height: mean (comment: A cell method)'
        actual = original.to_iris()
        actual_data = actual.core_data() if hasattr(actual, 'core_data') else actual.data
        assert_array_equal(actual_data, original.data)
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
        if hasattr(actual, 'core_data'):
            assert isinstance(original.data, type(actual.core_data()))
            assert isinstance(original.data, type(roundtripped.data))
        actual.remove_coord('time')
        auto_time_dimension = DataArray.from_iris(actual)
        assert auto_time_dimension.dims == ('distance', 'dim_1')

    @requires_iris
    @pytest.mark.parametrize('var_name, std_name, long_name, name, attrs', [('var_name', 'height', 'Height', 'var_name', {'standard_name': 'height', 'long_name': 'Height'}), (None, 'height', 'Height', 'height', {'standard_name': 'height', 'long_name': 'Height'}), (None, None, 'Height', 'Height', {'long_name': 'Height'}), (None, None, None, None, {})])
    def test_da_name_from_cube(self, std_name, long_name, var_name, name, attrs) -> None:
        from iris.cube import Cube
        cube = Cube([], var_name=var_name, standard_name=std_name, long_name=long_name)
        result = xr.DataArray.from_iris(cube)
        expected = xr.DataArray([], name=name, attrs=attrs)
        xr.testing.assert_identical(result, expected)

    @requires_iris
    @pytest.mark.parametrize('var_name, std_name, long_name, name, attrs', [('var_name', 'height', 'Height', 'var_name', {'standard_name': 'height', 'long_name': 'Height'}), (None, 'height', 'Height', 'height', {'standard_name': 'height', 'long_name': 'Height'}), (None, None, 'Height', 'Height', {'long_name': 'Height'}), (None, None, None, 'unknown', {})])
    def test_da_coord_name_from_cube(self, std_name, long_name, var_name, name, attrs) -> None:
        from iris.coords import DimCoord
        from iris.cube import Cube
        latitude = DimCoord([-90, 0, 90], standard_name=std_name, var_name=var_name, long_name=long_name)
        data = [0, 0, 0]
        cube = Cube(data, dim_coords_and_dims=[(latitude, 0)])
        result = xr.DataArray.from_iris(cube)
        expected = xr.DataArray(data, coords=[(name, [-90, 0, 90], attrs)])
        xr.testing.assert_identical(result, expected)

    @requires_iris
    def test_prevent_duplicate_coord_names(self) -> None:
        from iris.coords import DimCoord
        from iris.cube import Cube
        longitude = DimCoord([0, 360], standard_name='longitude', var_name='duplicate')
        latitude = DimCoord([-90, 0, 90], standard_name='latitude', var_name='duplicate')
        data = [[0, 0, 0], [0, 0, 0]]
        cube = Cube(data, dim_coords_and_dims=[(longitude, 0), (latitude, 1)])
        with pytest.raises(ValueError):
            xr.DataArray.from_iris(cube)

    @requires_iris
    @pytest.mark.parametrize('coord_values', [['IA', 'IL', 'IN'], [0, 2, 1]])
    def test_fallback_to_iris_AuxCoord(self, coord_values) -> None:
        from iris.coords import AuxCoord
        from iris.cube import Cube
        data = [0, 0, 0]
        da = xr.DataArray(data, coords=[coord_values], dims=['space'])
        result = xr.DataArray.to_iris(da)
        expected = Cube(data, aux_coords_and_dims=[(AuxCoord(coord_values, var_name='space'), 0)])
        assert result == expected