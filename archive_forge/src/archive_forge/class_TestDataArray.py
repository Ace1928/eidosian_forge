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
class TestDataArray:

    @pytest.mark.parametrize('variant', (pytest.param('with_dims', marks=pytest.mark.skip(reason="indexes don't support units")), 'with_coords', 'without_coords'))
    def test_init(self, variant, dtype):
        array = np.linspace(1, 2, 10, dtype=dtype) * unit_registry.m
        x = np.arange(len(array)) * unit_registry.s
        y = x.to(unit_registry.ms)
        variants = {'with_dims': {'x': x}, 'with_coords': {'y': ('x', y)}, 'without_coords': {}}
        kwargs = {'data': array, 'dims': 'x', 'coords': variants.get(variant)}
        data_array = xr.DataArray(**kwargs)
        assert isinstance(data_array.data, Quantity)
        assert all({name: isinstance(coord.data, Quantity) for name, coord in data_array.coords.items()}.values())

    @pytest.mark.parametrize('func', (pytest.param(str, id='str'), pytest.param(repr, id='repr')))
    @pytest.mark.parametrize('variant', (pytest.param('with_dims', marks=pytest.mark.skip(reason="indexes don't support units")), pytest.param('with_coords'), pytest.param('without_coords')))
    def test_repr(self, func, variant, dtype):
        array = np.linspace(1, 2, 10, dtype=dtype) * unit_registry.m
        x = np.arange(len(array)) * unit_registry.s
        y = x.to(unit_registry.ms)
        variants = {'with_dims': {'x': x}, 'with_coords': {'y': ('x', y)}, 'without_coords': {}}
        kwargs = {'data': array, 'dims': 'x', 'coords': variants.get(variant)}
        data_array = xr.DataArray(**kwargs)
        func(data_array)

    @pytest.mark.parametrize('func', (function('all'), function('any'), pytest.param(function('argmax'), marks=pytest.mark.skip(reason='calling np.argmax as a function on xarray objects is not supported')), pytest.param(function('argmin'), marks=pytest.mark.skip(reason='calling np.argmin as a function on xarray objects is not supported')), function('max'), function('mean'), pytest.param(function('median'), marks=pytest.mark.skip(reason='median does not work with dataarrays yet')), function('min'), function('prod'), function('sum'), function('std'), function('var'), function('cumsum'), function('cumprod'), method('all'), method('any'), method('argmax', dim='x'), method('argmin', dim='x'), method('max'), method('mean'), method('median'), method('min'), method('prod'), method('sum'), method('std'), method('var'), method('cumsum'), method('cumprod')), ids=repr)
    def test_aggregation(self, func, dtype):
        array = np.arange(10).astype(dtype) * (unit_registry.m if func.name != 'cumprod' else unit_registry.dimensionless)
        data_array = xr.DataArray(data=array, dims='x')
        numpy_kwargs = func.kwargs.copy()
        if 'dim' in numpy_kwargs:
            numpy_kwargs['axis'] = data_array.get_axis_num(numpy_kwargs.pop('dim'))
        units = extract_units(func(array))
        expected = attach_units(func(strip_units(data_array)), units)
        actual = func(data_array)
        assert_units_equal(expected, actual)
        assert_allclose(expected, actual)

    @pytest.mark.parametrize('func', (pytest.param(operator.neg, id='negate'), pytest.param(abs, id='absolute'), pytest.param(np.round, id='round')))
    def test_unary_operations(self, func, dtype):
        array = np.arange(10).astype(dtype) * unit_registry.m
        data_array = xr.DataArray(data=array)
        units = extract_units(func(array))
        expected = attach_units(func(strip_units(data_array)), units)
        actual = func(data_array)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('func', (pytest.param(lambda x: 2 * x, id='multiply'), pytest.param(lambda x: x + x, id='add'), pytest.param(lambda x: x[0] + x, id='add scalar'), pytest.param(lambda x: x.T @ x, id='matrix multiply')))
    def test_binary_operations(self, func, dtype):
        array = np.arange(10).astype(dtype) * unit_registry.m
        data_array = xr.DataArray(data=array)
        units = extract_units(func(array))
        with xr.set_options(use_opt_einsum=False):
            expected = attach_units(func(strip_units(data_array)), units)
            actual = func(data_array)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('comparison', (pytest.param(operator.lt, id='less_than'), pytest.param(operator.ge, id='greater_equal'), pytest.param(operator.eq, id='equal')))
    @pytest.mark.parametrize('unit,error', (pytest.param(1, ValueError, id='without_unit'), pytest.param(unit_registry.dimensionless, DimensionalityError, id='dimensionless'), pytest.param(unit_registry.s, DimensionalityError, id='incompatible_unit'), pytest.param(unit_registry.mm, None, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')))
    def test_comparison_operations(self, comparison, unit, error, dtype):
        array = np.array([10.1, 5.2, 6.5, 8.0, 21.3, 7.1, 1.3]).astype(dtype) * unit_registry.m
        data_array = xr.DataArray(data=array)
        value = 8
        to_compare_with = value * unit
        if error is not None and comparison is not operator.eq:
            with pytest.raises(error):
                comparison(array, to_compare_with)
            with pytest.raises(error):
                comparison(data_array, to_compare_with)
            return
        actual = comparison(data_array, to_compare_with)
        expected_units = {None: unit_registry.m if array.check(unit) else None}
        expected = array.check(unit) & comparison(strip_units(data_array), strip_units(convert_units(to_compare_with, expected_units)))
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('units,error', (pytest.param(unit_registry.dimensionless, None, id='dimensionless'), pytest.param(unit_registry.m, DimensionalityError, id='incompatible_unit'), pytest.param(unit_registry.degree, None, id='compatible_unit')))
    def test_univariate_ufunc(self, units, error, dtype):
        array = np.arange(10).astype(dtype) * units
        data_array = xr.DataArray(data=array)
        func = function('sin')
        if error is not None:
            with pytest.raises(error):
                np.sin(data_array)
            return
        expected = attach_units(func(strip_units(convert_units(data_array, {None: unit_registry.radians}))), {None: unit_registry.dimensionless})
        actual = func(data_array)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('unit,error', (pytest.param(1, DimensionalityError, id='without_unit'), pytest.param(unit_registry.dimensionless, DimensionalityError, id='dimensionless'), pytest.param(unit_registry.s, DimensionalityError, id='incompatible_unit'), pytest.param(unit_registry.mm, None, id='compatible_unit', marks=pytest.mark.xfail(reason='pint converts to the wrong units')), pytest.param(unit_registry.m, None, id='identical_unit')))
    def test_bivariate_ufunc(self, unit, error, dtype):
        original_unit = unit_registry.m
        array = np.arange(10).astype(dtype) * original_unit
        data_array = xr.DataArray(data=array)
        if error is not None:
            with pytest.raises(error):
                np.maximum(data_array, 1 * unit)
            return
        expected_units = {None: original_unit}
        expected = attach_units(np.maximum(strip_units(data_array), strip_units(convert_units(1 * unit, expected_units))), expected_units)
        actual = np.maximum(data_array, 1 * unit)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)
        actual = np.maximum(1 * unit, data_array)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('property', ('T', 'imag', 'real'))
    def test_numpy_properties(self, property, dtype):
        array = (np.arange(5 * 10).astype(dtype) + 1j * np.linspace(-1, 0, 5 * 10).astype(dtype)).reshape(5, 10) * unit_registry.s
        data_array = xr.DataArray(data=array, dims=('x', 'y'))
        expected = attach_units(getattr(strip_units(data_array), property), extract_units(data_array))
        actual = getattr(data_array, property)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('func', (method('conj'), method('argsort'), method('conjugate'), method('round')), ids=repr)
    def test_numpy_methods(self, func, dtype):
        array = np.arange(10).astype(dtype) * unit_registry.m
        data_array = xr.DataArray(data=array, dims='x')
        units = extract_units(func(array))
        expected = attach_units(strip_units(data_array), units)
        actual = func(data_array)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    def test_item(self, dtype):
        array = np.arange(10).astype(dtype) * unit_registry.m
        data_array = xr.DataArray(data=array)
        func = method('item', 2)
        expected = func(strip_units(data_array)) * unit_registry.m
        actual = func(data_array)
        assert_duckarray_allclose(expected, actual)

    @pytest.mark.parametrize('unit,error', (pytest.param(1, DimensionalityError, id='no_unit'), pytest.param(unit_registry.dimensionless, DimensionalityError, id='dimensionless'), pytest.param(unit_registry.s, DimensionalityError, id='incompatible_unit'), pytest.param(unit_registry.cm, None, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')))
    @pytest.mark.parametrize('func', (method('searchsorted', 5), pytest.param(function('searchsorted', 5), marks=pytest.mark.xfail(reason='xarray does not implement __array_function__'))), ids=repr)
    def test_searchsorted(self, func, unit, error, dtype):
        array = np.arange(10).astype(dtype) * unit_registry.m
        data_array = xr.DataArray(data=array)
        scalar_types = (int, float)
        args = list((value * unit for value in func.args))
        kwargs = {key: value * unit if isinstance(value, scalar_types) else value for key, value in func.kwargs.items()}
        if error is not None:
            with pytest.raises(error):
                func(data_array, *args, **kwargs)
            return
        units = extract_units(data_array)
        expected_units = extract_units(func(array, *args, **kwargs))
        stripped_args = [strip_units(convert_units(value, units)) for value in args]
        stripped_kwargs = {key: strip_units(convert_units(value, units)) for key, value in kwargs.items()}
        expected = attach_units(func(strip_units(data_array), *stripped_args, **stripped_kwargs), expected_units)
        actual = func(data_array, *args, **kwargs)
        assert_units_equal(expected, actual)
        np.testing.assert_allclose(expected, actual)

    @pytest.mark.parametrize('func', (method('clip', min=3, max=8), pytest.param(function('clip', a_min=3, a_max=8), marks=pytest.mark.xfail(reason='xarray does not implement __array_function__'))), ids=repr)
    @pytest.mark.parametrize('unit,error', (pytest.param(1, DimensionalityError, id='no_unit'), pytest.param(unit_registry.dimensionless, DimensionalityError, id='dimensionless'), pytest.param(unit_registry.s, DimensionalityError, id='incompatible_unit'), pytest.param(unit_registry.cm, None, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')))
    def test_numpy_methods_with_args(self, func, unit, error, dtype):
        array = np.arange(10).astype(dtype) * unit_registry.m
        data_array = xr.DataArray(data=array)
        scalar_types = (int, float)
        args = list((value * unit for value in func.args))
        kwargs = {key: value * unit if isinstance(value, scalar_types) else value for key, value in func.kwargs.items()}
        if error is not None:
            with pytest.raises(error):
                func(data_array, *args, **kwargs)
            return
        units = extract_units(data_array)
        expected_units = extract_units(func(array, *args, **kwargs))
        stripped_args = [strip_units(convert_units(value, units)) for value in args]
        stripped_kwargs = {key: strip_units(convert_units(value, units)) for key, value in kwargs.items()}
        expected = attach_units(func(strip_units(data_array), *stripped_args, **stripped_kwargs), expected_units)
        actual = func(data_array, *args, **kwargs)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('func', (method('isnull'), method('notnull'), method('count')), ids=repr)
    def test_missing_value_detection(self, func, dtype):
        array = np.array([[1.4, 2.3, np.nan, 7.2], [np.nan, 9.7, np.nan, np.nan], [2.1, np.nan, np.nan, 4.6], [9.9, np.nan, 7.2, 9.1]]) * unit_registry.degK
        data_array = xr.DataArray(data=array)
        expected = func(strip_units(data_array))
        actual = func(data_array)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.xfail(reason='ffill and bfill lose units in data')
    @pytest.mark.parametrize('func', (method('ffill'), method('bfill')), ids=repr)
    def test_missing_value_filling(self, func, dtype):
        array = np.array([1.4, np.nan, 2.3, np.nan, np.nan, 9.1]).astype(dtype) * unit_registry.degK
        x = np.arange(len(array))
        data_array = xr.DataArray(data=array, coords={'x': x}, dims='x')
        expected = attach_units(func(strip_units(data_array), dim='x'), extract_units(data_array))
        actual = func(data_array, dim='x')
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('unit,error', (pytest.param(1, DimensionalityError, id='no_unit'), pytest.param(unit_registry.dimensionless, DimensionalityError, id='dimensionless'), pytest.param(unit_registry.s, DimensionalityError, id='incompatible_unit'), pytest.param(unit_registry.cm, None, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')))
    @pytest.mark.parametrize('fill_value', (pytest.param(-1, id='python_scalar'), pytest.param(np.array(-1), id='numpy_scalar'), pytest.param(np.array([-1]), id='numpy_array')))
    def test_fillna(self, fill_value, unit, error, dtype):
        original_unit = unit_registry.m
        array = np.array([1.4, np.nan, 2.3, np.nan, np.nan, 9.1]).astype(dtype) * original_unit
        data_array = xr.DataArray(data=array)
        func = method('fillna')
        value = fill_value * unit
        if error is not None:
            with pytest.raises(error):
                func(data_array, value=value)
            return
        units = extract_units(data_array)
        expected = attach_units(func(strip_units(data_array), value=strip_units(convert_units(value, units))), units)
        actual = func(data_array, value=value)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    def test_dropna(self, dtype):
        array = np.array([1.4, np.nan, 2.3, np.nan, np.nan, 9.1]).astype(dtype) * unit_registry.m
        x = np.arange(len(array))
        data_array = xr.DataArray(data=array, coords={'x': x}, dims=['x'])
        units = extract_units(data_array)
        expected = attach_units(strip_units(data_array).dropna(dim='x'), units)
        actual = data_array.dropna(dim='x')
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('unit', (pytest.param(1, id='no_unit'), pytest.param(unit_registry.dimensionless, id='dimensionless'), pytest.param(unit_registry.s, id='incompatible_unit'), pytest.param(unit_registry.cm, id='compatible_unit'), pytest.param(unit_registry.m, id='identical_unit')))
    def test_isin(self, unit, dtype):
        array = np.array([1.4, np.nan, 2.3, np.nan, np.nan, 9.1]).astype(dtype) * unit_registry.m
        data_array = xr.DataArray(data=array, dims='x')
        raw_values = np.array([1.4, np.nan, 2.3]).astype(dtype)
        values = raw_values * unit
        units = {None: unit_registry.m if array.check(unit) else None}
        expected = strip_units(data_array).isin(strip_units(convert_units(values, units))) & array.check(unit)
        actual = data_array.isin(values)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('variant', ('masking', 'replacing_scalar', 'replacing_array', 'dropping'))
    @pytest.mark.parametrize('unit,error', (pytest.param(1, DimensionalityError, id='no_unit'), pytest.param(unit_registry.dimensionless, DimensionalityError, id='dimensionless'), pytest.param(unit_registry.s, DimensionalityError, id='incompatible_unit'), pytest.param(unit_registry.cm, None, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')))
    def test_where(self, variant, unit, error, dtype):
        original_unit = unit_registry.m
        array = np.linspace(0, 1, 10).astype(dtype) * original_unit
        data_array = xr.DataArray(data=array)
        condition = data_array < 0.5 * original_unit
        other = np.linspace(-2, -1, 10).astype(dtype) * unit
        variant_kwargs = {'masking': {'cond': condition}, 'replacing_scalar': {'cond': condition, 'other': -1 * unit}, 'replacing_array': {'cond': condition, 'other': other}, 'dropping': {'cond': condition, 'drop': True}}
        kwargs = variant_kwargs.get(variant)
        kwargs_without_units = {key: strip_units(convert_units(value, {None: original_unit if array.check(unit) else None})) for key, value in kwargs.items()}
        if variant not in ('masking', 'dropping') and error is not None:
            with pytest.raises(error):
                data_array.where(**kwargs)
            return
        expected = attach_units(strip_units(data_array).where(**kwargs_without_units), extract_units(data_array))
        actual = data_array.where(**kwargs)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.xfail(reason='uses numpy.vectorize')
    def test_interpolate_na(self):
        array = np.array([-1.03, 0.1, 1.4, np.nan, 2.3, np.nan, np.nan, 9.1]) * unit_registry.m
        x = np.arange(len(array))
        data_array = xr.DataArray(data=array, coords={'x': x}, dims='x')
        units = extract_units(data_array)
        expected = attach_units(strip_units(data_array).interpolate_na(dim='x'), units)
        actual = data_array.interpolate_na(dim='x')
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('unit,error', (pytest.param(1, DimensionalityError, id='no_unit'), pytest.param(unit_registry.dimensionless, DimensionalityError, id='dimensionless'), pytest.param(unit_registry.s, DimensionalityError, id='incompatible_unit'), pytest.param(unit_registry.cm, None, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')))
    def test_combine_first(self, unit, error, dtype):
        array = np.zeros(shape=(2, 2), dtype=dtype) * unit_registry.m
        other_array = np.ones_like(array) * unit
        data_array = xr.DataArray(data=array, coords={'x': ['a', 'b'], 'y': [-1, 0]}, dims=['x', 'y'])
        other = xr.DataArray(data=other_array, coords={'x': ['b', 'c'], 'y': [0, 1]}, dims=['x', 'y'])
        if error is not None:
            with pytest.raises(error):
                data_array.combine_first(other)
            return
        units = extract_units(data_array)
        expected = attach_units(strip_units(data_array).combine_first(strip_units(convert_units(other, units))), units)
        actual = data_array.combine_first(other)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('unit', (pytest.param(1, id='no_unit'), pytest.param(unit_registry.dimensionless, id='dimensionless'), pytest.param(unit_registry.s, id='incompatible_unit'), pytest.param(unit_registry.cm, id='compatible_unit'), pytest.param(unit_registry.m, id='identical_unit')))
    @pytest.mark.parametrize('variation', ('data', pytest.param('dims', marks=pytest.mark.skip(reason="indexes don't support units")), 'coords'))
    @pytest.mark.parametrize('func', (method('equals'), pytest.param(method('identical'), marks=pytest.mark.skip(reason='the behavior of identical is undecided'))), ids=repr)
    def test_comparisons(self, func, variation, unit, dtype):

        def is_compatible(a, b):
            a = a if a is not None else 1
            b = b if b is not None else 1
            quantity = np.arange(5) * a
            return a == b or quantity.check(b)
        data = np.linspace(0, 5, 10).astype(dtype)
        coord = np.arange(len(data)).astype(dtype)
        base_unit = unit_registry.m
        array = data * (base_unit if variation == 'data' else 1)
        x = coord * (base_unit if variation == 'dims' else 1)
        y = coord * (base_unit if variation == 'coords' else 1)
        variations = {'data': (unit, 1, 1), 'dims': (1, unit, 1), 'coords': (1, 1, unit)}
        data_unit, dim_unit, coord_unit = variations.get(variation)
        data_array = xr.DataArray(data=array, coords={'x': x, 'y': ('x', y)}, dims='x')
        other = attach_units(strip_units(data_array), {None: data_unit, 'x': dim_unit, 'y': coord_unit})
        units = extract_units(data_array)
        other_units = extract_units(other)
        equal_arrays = all((is_compatible(units[name], other_units[name]) for name in units.keys())) and strip_units(data_array).equals(strip_units(convert_units(other, extract_units(data_array))))
        equal_units = units == other_units
        expected = equal_arrays and (func.name != 'identical' or equal_units)
        actual = func(data_array, other)
        assert expected == actual

    @pytest.mark.parametrize('unit', (pytest.param(1, id='no_unit'), pytest.param(unit_registry.dimensionless, id='dimensionless'), pytest.param(unit_registry.s, id='incompatible_unit'), pytest.param(unit_registry.cm, id='compatible_unit'), pytest.param(unit_registry.m, id='identical_unit')))
    @pytest.mark.parametrize('variant', ('data', pytest.param('dims', marks=pytest.mark.skip(reason="indexes don't support units")), 'coords'))
    def test_broadcast_like(self, variant, unit, dtype):
        original_unit = unit_registry.m
        variants = {'data': ((original_unit, unit), (1, 1), (1, 1)), 'dims': ((1, 1), (original_unit, unit), (1, 1)), 'coords': ((1, 1), (1, 1), (original_unit, unit))}
        (data_unit1, data_unit2), (dim_unit1, dim_unit2), (coord_unit1, coord_unit2) = variants.get(variant)
        array1 = np.linspace(1, 2, 2 * 1).reshape(2, 1).astype(dtype) * data_unit1
        array2 = np.linspace(0, 1, 2 * 3).reshape(2, 3).astype(dtype) * data_unit2
        x1 = np.arange(2) * dim_unit1
        x2 = np.arange(2) * dim_unit2
        y1 = np.array([0]) * dim_unit1
        y2 = np.arange(3) * dim_unit2
        u1 = np.linspace(0, 1, 2) * coord_unit1
        u2 = np.linspace(0, 1, 2) * coord_unit2
        arr1 = xr.DataArray(data=array1, coords={'x': x1, 'y': y1, 'u': ('x', u1)}, dims=('x', 'y'))
        arr2 = xr.DataArray(data=array2, coords={'x': x2, 'y': y2, 'u': ('x', u2)}, dims=('x', 'y'))
        expected = attach_units(strip_units(arr1).broadcast_like(strip_units(arr2)), extract_units(arr1))
        actual = arr1.broadcast_like(arr2)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('unit', (pytest.param(1, id='no_unit'), pytest.param(unit_registry.dimensionless, id='dimensionless'), pytest.param(unit_registry.s, id='incompatible_unit'), pytest.param(unit_registry.cm, id='compatible_unit'), pytest.param(unit_registry.m, id='identical_unit')))
    def test_broadcast_equals(self, unit, dtype):
        left_array = np.ones(shape=(2, 2), dtype=dtype) * unit_registry.m
        right_array = np.ones(shape=(2,), dtype=dtype) * unit
        left = xr.DataArray(data=left_array, dims=('x', 'y'))
        right = xr.DataArray(data=right_array, dims='x')
        units = {**extract_units(left), **({} if left_array.check(unit) else {None: None})}
        expected = strip_units(left).broadcast_equals(strip_units(convert_units(right, units))) & left_array.check(unit)
        actual = left.broadcast_equals(right)
        assert expected == actual

    def test_pad(self, dtype):
        array = np.linspace(0, 5, 10).astype(dtype) * unit_registry.m
        data_array = xr.DataArray(data=array, dims='x')
        units = extract_units(data_array)
        expected = attach_units(strip_units(data_array).pad(x=(2, 3)), units)
        actual = data_array.pad(x=(2, 3))
        assert_units_equal(expected, actual)
        assert_equal(expected, actual)

    @pytest.mark.parametrize('variant', ('data', pytest.param('dims', marks=pytest.mark.skip(reason="indexes don't support units")), 'coords'))
    @pytest.mark.parametrize('func', (method('pipe', lambda da: da * 10), method('assign_coords', w=('y', np.arange(10) * unit_registry.mm)), method('assign_attrs', attr1='value'), method('rename', u='v'), pytest.param(method('swap_dims', {'x': 'u'}), marks=pytest.mark.skip(reason="indexes don't support units")), pytest.param(method('expand_dims', dim={'z': np.linspace(10, 20, 12) * unit_registry.s}, axis=1), marks=pytest.mark.skip(reason="indexes don't support units")), method('drop_vars', 'x'), method('reset_coords', names='u'), method('copy'), method('astype', np.float32)), ids=repr)
    def test_content_manipulation(self, func, variant, dtype):
        unit = unit_registry.m
        variants = {'data': (unit, 1, 1), 'dims': (1, unit, 1), 'coords': (1, 1, unit)}
        data_unit, dim_unit, coord_unit = variants.get(variant)
        quantity = np.linspace(0, 10, 5 * 10).reshape(5, 10).astype(dtype) * data_unit
        x = np.arange(quantity.shape[0]) * dim_unit
        y = np.arange(quantity.shape[1]) * dim_unit
        u = np.linspace(0, 1, quantity.shape[0]) * coord_unit
        data_array = xr.DataArray(name='a', data=quantity, coords={'x': x, 'u': ('x', u), 'y': y}, dims=('x', 'y'))
        stripped_kwargs = {key: array_strip_units(value) for key, value in func.kwargs.items()}
        units = extract_units(data_array)
        units['u'] = getattr(u, 'units', None)
        units['v'] = getattr(u, 'units', None)
        expected = attach_units(func(strip_units(data_array), **stripped_kwargs), units)
        actual = func(data_array)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('unit', (pytest.param(1, id='no_unit'), pytest.param(unit_registry.dimensionless, id='dimensionless'), pytest.param(unit_registry.degK, id='with_unit')))
    def test_copy(self, unit, dtype):
        quantity = np.linspace(0, 10, 20, dtype=dtype) * unit_registry.pascal
        new_data = np.arange(20)
        data_array = xr.DataArray(data=quantity, dims='x')
        expected = attach_units(strip_units(data_array).copy(data=new_data), {None: unit})
        actual = data_array.copy(data=new_data * unit)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('indices', (pytest.param(4, id='single index'), pytest.param([5, 2, 9, 1], id='multiple indices')))
    def test_isel(self, indices, dtype):
        array = np.arange(10).astype(dtype) * unit_registry.s
        data_array = xr.DataArray(data=array, dims='x')
        expected = attach_units(strip_units(data_array).isel(x=indices), extract_units(data_array))
        actual = data_array.isel(x=indices)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.skip(reason="indexes don't support units")
    @pytest.mark.parametrize('raw_values', (pytest.param(10, id='single_value'), pytest.param([10, 5, 13], id='list_of_values'), pytest.param(np.array([9, 3, 7, 12]), id='array_of_values')))
    @pytest.mark.parametrize('unit,error', (pytest.param(1, KeyError, id='no_units'), pytest.param(unit_registry.dimensionless, KeyError, id='dimensionless'), pytest.param(unit_registry.degree, KeyError, id='incompatible_unit'), pytest.param(unit_registry.dm, KeyError, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')))
    def test_sel(self, raw_values, unit, error, dtype):
        array = np.linspace(5, 10, 20).astype(dtype) * unit_registry.m
        x = np.arange(len(array)) * unit_registry.m
        data_array = xr.DataArray(data=array, coords={'x': x}, dims='x')
        values = raw_values * unit
        if error is not None and (not (isinstance(raw_values, (int, float)) and x.check(unit))):
            with pytest.raises(error):
                data_array.sel(x=values)
            return
        expected = attach_units(strip_units(data_array).sel(x=strip_units(convert_units(values, {None: array.units}))), extract_units(data_array))
        actual = data_array.sel(x=values)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.skip(reason="indexes don't support units")
    @pytest.mark.parametrize('raw_values', (pytest.param(10, id='single_value'), pytest.param([10, 5, 13], id='list_of_values'), pytest.param(np.array([9, 3, 7, 12]), id='array_of_values')))
    @pytest.mark.parametrize('unit,error', (pytest.param(1, KeyError, id='no_units'), pytest.param(unit_registry.dimensionless, KeyError, id='dimensionless'), pytest.param(unit_registry.degree, KeyError, id='incompatible_unit'), pytest.param(unit_registry.dm, KeyError, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')))
    def test_loc(self, raw_values, unit, error, dtype):
        array = np.linspace(5, 10, 20).astype(dtype) * unit_registry.m
        x = np.arange(len(array)) * unit_registry.m
        data_array = xr.DataArray(data=array, coords={'x': x}, dims='x')
        values = raw_values * unit
        if error is not None and (not (isinstance(raw_values, (int, float)) and x.check(unit))):
            with pytest.raises(error):
                data_array.loc[{'x': values}]
            return
        expected = attach_units(strip_units(data_array).loc[{'x': strip_units(convert_units(values, {None: array.units}))}], extract_units(data_array))
        actual = data_array.loc[{'x': values}]
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.skip(reason="indexes don't support units")
    @pytest.mark.parametrize('raw_values', (pytest.param(10, id='single_value'), pytest.param([10, 5, 13], id='list_of_values'), pytest.param(np.array([9, 3, 7, 12]), id='array_of_values')))
    @pytest.mark.parametrize('unit,error', (pytest.param(1, KeyError, id='no_units'), pytest.param(unit_registry.dimensionless, KeyError, id='dimensionless'), pytest.param(unit_registry.degree, KeyError, id='incompatible_unit'), pytest.param(unit_registry.dm, KeyError, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')))
    def test_drop_sel(self, raw_values, unit, error, dtype):
        array = np.linspace(5, 10, 20).astype(dtype) * unit_registry.m
        x = np.arange(len(array)) * unit_registry.m
        data_array = xr.DataArray(data=array, coords={'x': x}, dims='x')
        values = raw_values * unit
        if error is not None and (not (isinstance(raw_values, (int, float)) and x.check(unit))):
            with pytest.raises(error):
                data_array.drop_sel(x=values)
            return
        expected = attach_units(strip_units(data_array).drop_sel(x=strip_units(convert_units(values, {None: x.units}))), extract_units(data_array))
        actual = data_array.drop_sel(x=values)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('dim', ('x', 'y', 'z', 't', 'all'))
    @pytest.mark.parametrize('shape', (pytest.param((10, 20), id='nothing_squeezable'), pytest.param((10, 20, 1), id='last_dimension_squeezable'), pytest.param((10, 1, 20), id='middle_dimension_squeezable'), pytest.param((1, 10, 20), id='first_dimension_squeezable'), pytest.param((1, 10, 1, 20), id='first_and_last_dimension_squeezable')))
    def test_squeeze(self, shape, dim, dtype):
        names = 'xyzt'
        dim_lengths = dict(zip(names, shape))
        names = 'xyzt'
        array = np.arange(10 * 20).astype(dtype).reshape(shape) * unit_registry.J
        data_array = xr.DataArray(data=array, dims=tuple(names[:len(shape)]))
        kwargs = {'dim': dim} if dim != 'all' and dim_lengths.get(dim, 0) == 1 else {}
        expected = attach_units(strip_units(data_array).squeeze(**kwargs), extract_units(data_array))
        actual = data_array.squeeze(**kwargs)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('func', (method('head', x=7, y=3), method('tail', x=7, y=3), method('thin', x=7, y=3)), ids=repr)
    def test_head_tail_thin(self, func, dtype):
        array = np.linspace(1, 2, 10 * 5).reshape(10, 5) * unit_registry.degK
        data_array = xr.DataArray(data=array, dims=('x', 'y'))
        expected = attach_units(func(strip_units(data_array)), extract_units(data_array))
        actual = func(data_array)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('variant', ('data', 'coords'))
    @pytest.mark.parametrize('func', (pytest.param(method('interp'), marks=pytest.mark.xfail(reason='uses scipy')), method('reindex')), ids=repr)
    def test_interp_reindex(self, variant, func, dtype):
        variants = {'data': (unit_registry.m, 1), 'coords': (1, unit_registry.m)}
        data_unit, coord_unit = variants.get(variant)
        array = np.linspace(1, 2, 10).astype(dtype) * data_unit
        y = np.arange(10) * coord_unit
        x = np.arange(10)
        new_x = np.arange(10) + 0.5
        data_array = xr.DataArray(array, coords={'x': x, 'y': ('x', y)}, dims='x')
        units = extract_units(data_array)
        expected = attach_units(func(strip_units(data_array), x=new_x), units)
        actual = func(data_array, x=new_x)
        assert_units_equal(expected, actual)
        assert_allclose(expected, actual)

    @pytest.mark.skip(reason="indexes don't support units")
    @pytest.mark.parametrize('unit,error', (pytest.param(1, DimensionalityError, id='no_unit'), pytest.param(unit_registry.dimensionless, DimensionalityError, id='dimensionless'), pytest.param(unit_registry.s, DimensionalityError, id='incompatible_unit'), pytest.param(unit_registry.cm, None, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')))
    @pytest.mark.parametrize('func', (method('interp'), method('reindex')), ids=repr)
    def test_interp_reindex_indexing(self, func, unit, error, dtype):
        array = np.linspace(1, 2, 10).astype(dtype)
        x = np.arange(10) * unit_registry.m
        new_x = (np.arange(10) + 0.5) * unit
        data_array = xr.DataArray(array, coords={'x': x}, dims='x')
        if error is not None:
            with pytest.raises(error):
                func(data_array, x=new_x)
            return
        units = extract_units(data_array)
        expected = attach_units(func(strip_units(data_array), x=strip_units(convert_units(new_x, {None: unit_registry.m}))), units)
        actual = func(data_array, x=new_x)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('variant', ('data', 'coords'))
    @pytest.mark.parametrize('func', (pytest.param(method('interp_like'), marks=pytest.mark.xfail(reason='uses scipy')), method('reindex_like')), ids=repr)
    def test_interp_reindex_like(self, variant, func, dtype):
        variants = {'data': (unit_registry.m, 1), 'coords': (1, unit_registry.m)}
        data_unit, coord_unit = variants.get(variant)
        array = np.linspace(1, 2, 10).astype(dtype) * data_unit
        coord = np.arange(10) * coord_unit
        x = np.arange(10)
        new_x = np.arange(-2, 2) + 0.5
        data_array = xr.DataArray(array, coords={'x': x, 'y': ('x', coord)}, dims='x')
        other = xr.DataArray(np.empty_like(new_x), coords={'x': new_x}, dims='x')
        units = extract_units(data_array)
        expected = attach_units(func(strip_units(data_array), other), units)
        actual = func(data_array, other)
        assert_units_equal(expected, actual)
        assert_allclose(expected, actual)

    @pytest.mark.skip(reason="indexes don't support units")
    @pytest.mark.parametrize('unit,error', (pytest.param(1, DimensionalityError, id='no_unit'), pytest.param(unit_registry.dimensionless, DimensionalityError, id='dimensionless'), pytest.param(unit_registry.s, DimensionalityError, id='incompatible_unit'), pytest.param(unit_registry.cm, None, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')))
    @pytest.mark.parametrize('func', (method('interp_like'), method('reindex_like')), ids=repr)
    def test_interp_reindex_like_indexing(self, func, unit, error, dtype):
        array = np.linspace(1, 2, 10).astype(dtype)
        x = np.arange(10) * unit_registry.m
        new_x = (np.arange(-2, 2) + 0.5) * unit
        data_array = xr.DataArray(array, coords={'x': x}, dims='x')
        other = xr.DataArray(np.empty_like(new_x), {'x': new_x}, dims='x')
        if error is not None:
            with pytest.raises(error):
                func(data_array, other)
            return
        units = extract_units(data_array)
        expected = attach_units(func(strip_units(data_array), strip_units(convert_units(other, {None: unit_registry.m}))), units)
        actual = func(data_array, other)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('func', (method('unstack'), method('reset_index', 'z'), method('reorder_levels')), ids=repr)
    def test_stacking_stacked(self, func, dtype):
        array = np.linspace(0, 10, 5 * 10).reshape(5, 10).astype(dtype) * unit_registry.m
        x = np.arange(array.shape[0])
        y = np.arange(array.shape[1])
        data_array = xr.DataArray(name='data', data=array, coords={'x': x, 'y': y}, dims=('x', 'y'))
        stacked = data_array.stack(z=('x', 'y'))
        expected = attach_units(func(strip_units(stacked)), {'data': unit_registry.m})
        actual = func(stacked)
        assert_units_equal(expected, actual)
        if func.name == 'reset_index':
            assert_identical(expected, actual, check_default_indexes=False)
        else:
            assert_identical(expected, actual)

    @pytest.mark.skip(reason="indexes don't support units")
    def test_to_unstacked_dataset(self, dtype):
        array = np.linspace(0, 10, 5 * 10).reshape(5, 10).astype(dtype) * unit_registry.pascal
        x = np.arange(array.shape[0]) * unit_registry.m
        y = np.arange(array.shape[1]) * unit_registry.s
        data_array = xr.DataArray(data=array, coords={'x': x, 'y': y}, dims=('x', 'y')).stack(z=('x', 'y'))
        func = method('to_unstacked_dataset', dim='z')
        expected = attach_units(func(strip_units(data_array)), {'y': y.units, **dict(zip(x.magnitude, [array.units] * len(y)))}).rename({elem.magnitude: elem for elem in x})
        actual = func(data_array)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('func', (method('transpose', 'y', 'x', 'z'), method('stack', a=('x', 'y')), method('set_index', x='x2'), method('shift', x=2), pytest.param(method('rank', dim='x'), marks=pytest.mark.skip(reason='rank not implemented for non-ndarray')), method('roll', x=2, roll_coords=False), method('sortby', 'x2')), ids=repr)
    def test_stacking_reordering(self, func, dtype):
        array = np.linspace(0, 10, 2 * 5 * 10).reshape(2, 5, 10).astype(dtype) * unit_registry.m
        x = np.arange(array.shape[0])
        y = np.arange(array.shape[1])
        z = np.arange(array.shape[2])
        x2 = np.linspace(0, 1, array.shape[0])[::-1]
        data_array = xr.DataArray(name='data', data=array, coords={'x': x, 'y': y, 'z': z, 'x2': ('x', x2)}, dims=('x', 'y', 'z'))
        expected = attach_units(func(strip_units(data_array)), {None: unit_registry.m})
        actual = func(data_array)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('variant', (pytest.param('dims', marks=pytest.mark.skip(reason="indexes don't support units")), 'coords'))
    @pytest.mark.parametrize('func', (method('differentiate', fallback_func=np.gradient), method('integrate', fallback_func=duck_array_ops.cumulative_trapezoid), method('cumulative_integrate', fallback_func=duck_array_ops.trapz)), ids=repr)
    def test_differentiate_integrate(self, func, variant, dtype):
        data_unit = unit_registry.m
        unit = unit_registry.s
        variants = {'dims': ('x', unit, 1), 'coords': ('u', 1, unit)}
        coord, dim_unit, coord_unit = variants.get(variant)
        array = np.linspace(0, 10, 5 * 10).reshape(5, 10).astype(dtype) * data_unit
        x = np.arange(array.shape[0]) * dim_unit
        y = np.arange(array.shape[1]) * dim_unit
        u = np.linspace(0, 1, array.shape[0]) * coord_unit
        data_array = xr.DataArray(data=array, coords={'x': x, 'y': y, 'u': ('x', u)}, dims=('x', 'y'))
        units = extract_units(data_array)
        units.update(extract_units(func(data_array.data, getattr(data_array, coord).data, axis=0)))
        expected = attach_units(func(strip_units(data_array), coord=strip_units(coord)), units)
        actual = func(data_array, coord=coord)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('compute_backend', ['numbagg', None], indirect=True)
    @pytest.mark.parametrize('variant', ('data', pytest.param('dims', marks=pytest.mark.skip(reason="indexes don't support units")), 'coords'))
    @pytest.mark.parametrize('func', (method('diff', dim='x'), method('quantile', q=[0.25, 0.75]), method('reduce', func=np.sum, dim='x'), pytest.param(lambda x: x.dot(x), id='method_dot')), ids=repr)
    def test_computation(self, func, variant, dtype, compute_backend):
        unit = unit_registry.m
        variants = {'data': (unit, 1, 1), 'dims': (1, unit, 1), 'coords': (1, 1, unit)}
        data_unit, dim_unit, coord_unit = variants.get(variant)
        array = np.linspace(0, 10, 5 * 10).reshape(5, 10).astype(dtype) * data_unit
        x = np.arange(array.shape[0]) * dim_unit
        y = np.arange(array.shape[1]) * dim_unit
        u = np.linspace(0, 1, array.shape[0]) * coord_unit
        data_array = xr.DataArray(data=array, coords={'x': x, 'y': y, 'u': ('x', u)}, dims=('x', 'y'))
        units = extract_units(data_array)
        if not isinstance(func, (function, method)):
            units.update(extract_units(func(array.reshape(-1))))
        with xr.set_options(use_opt_einsum=False):
            expected = attach_units(func(strip_units(data_array)), units)
            actual = func(data_array)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('variant', ('data', pytest.param('dims', marks=pytest.mark.skip(reason="indexes don't support units")), 'coords'))
    @pytest.mark.parametrize('func', (method('groupby', 'x'), method('groupby_bins', 'y', bins=4), method('coarsen', y=2), method('rolling', y=3), pytest.param(method('rolling_exp', y=3), marks=requires_numbagg), method('weighted', xr.DataArray(data=np.linspace(0, 1, 10), dims='y'))), ids=repr)
    def test_computation_objects(self, func, variant, dtype):
        if variant == 'data':
            if func.name == 'rolling_exp':
                pytest.xfail(reason='numbagg functions are not supported by pint')
            elif func.name == 'rolling':
                pytest.xfail(reason='numpy.lib.stride_tricks.as_strided converts to ndarray')
        unit = unit_registry.m
        variants = {'data': (unit, 1, 1), 'dims': (1, unit, 1), 'coords': (1, 1, unit)}
        data_unit, dim_unit, coord_unit = variants.get(variant)
        array = np.linspace(0, 10, 5 * 10).reshape(5, 10).astype(dtype) * data_unit
        x = np.array([0, 0, 1, 2, 2]) * dim_unit
        y = np.arange(array.shape[1]) * 3 * dim_unit
        u = np.linspace(0, 1, 5) * coord_unit
        data_array = xr.DataArray(data=array, coords={'x': x, 'y': y, 'u': ('x', u)}, dims=('x', 'y'))
        units = extract_units(data_array)
        expected = attach_units(func(strip_units(data_array)).mean(), units)
        actual = func(data_array).mean()
        assert_units_equal(expected, actual)
        assert_allclose(expected, actual)

    def test_resample(self, dtype):
        array = np.linspace(0, 5, 10).astype(dtype) * unit_registry.m
        time = xr.date_range('10-09-2010', periods=len(array), freq='YE')
        data_array = xr.DataArray(data=array, coords={'time': time}, dims='time')
        units = extract_units(data_array)
        func = method('resample', time='6ME')
        expected = attach_units(func(strip_units(data_array)).mean(), units)
        actual = func(data_array).mean()
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('compute_backend', ['numbagg', None], indirect=True)
    @pytest.mark.parametrize('variant', ('data', pytest.param('dims', marks=pytest.mark.skip(reason="indexes don't support units")), 'coords'))
    @pytest.mark.parametrize('func', (method('assign_coords', z=('x', np.arange(5) * unit_registry.s)), method('first'), method('last'), method('quantile', q=[0.25, 0.5, 0.75], dim='x')), ids=repr)
    def test_grouped_operations(self, func, variant, dtype, compute_backend):
        unit = unit_registry.m
        variants = {'data': (unit, 1, 1), 'dims': (1, unit, 1), 'coords': (1, 1, unit)}
        data_unit, dim_unit, coord_unit = variants.get(variant)
        array = np.linspace(0, 10, 5 * 10).reshape(5, 10).astype(dtype) * data_unit
        x = np.arange(array.shape[0]) * dim_unit
        y = np.arange(array.shape[1]) * 3 * dim_unit
        u = np.linspace(0, 1, array.shape[0]) * coord_unit
        data_array = xr.DataArray(data=array, coords={'x': x, 'y': y, 'u': ('x', u)}, dims=('x', 'y'))
        units = {**extract_units(data_array), **{'z': unit_registry.s, 'q': None}}
        stripped_kwargs = {key: strip_units(value) if not isinstance(value, tuple) else tuple((strip_units(elem) for elem in value)) for key, value in func.kwargs.items()}
        expected = attach_units(func(strip_units(data_array).groupby('y', squeeze=False), **stripped_kwargs), units)
        actual = func(data_array.groupby('y', squeeze=False))
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)