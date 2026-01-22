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
class TestVariable:

    @pytest.mark.parametrize('func', (method('all'), method('any'), method('argmax', dim='x'), method('argmin', dim='x'), method('argsort'), method('cumprod'), method('cumsum'), method('max'), method('mean'), method('median'), method('min'), method('prod'), method('std'), method('sum'), method('var')), ids=repr)
    def test_aggregation(self, func, dtype):
        array = np.linspace(0, 1, 10).astype(dtype) * (unit_registry.m if func.name != 'cumprod' else unit_registry.dimensionless)
        variable = xr.Variable('x', array)
        numpy_kwargs = func.kwargs.copy()
        if 'dim' in func.kwargs:
            numpy_kwargs['axis'] = variable.get_axis_num(numpy_kwargs.pop('dim'))
        units = extract_units(func(array, **numpy_kwargs))
        expected = attach_units(func(strip_units(variable)), units)
        actual = func(variable)
        assert_units_equal(expected, actual)
        assert_allclose(expected, actual)

    def test_aggregate_complex(self):
        variable = xr.Variable('x', [1, 2j, np.nan] * unit_registry.m)
        expected = xr.Variable((), (0.5 + 1j) * unit_registry.m)
        actual = variable.mean()
        assert_units_equal(expected, actual)
        assert_allclose(expected, actual)

    @pytest.mark.parametrize('func', (method('astype', np.float32), method('conj'), method('conjugate'), method('clip', min=2, max=7)), ids=repr)
    @pytest.mark.parametrize('unit,error', (pytest.param(1, DimensionalityError, id='no_unit'), pytest.param(unit_registry.dimensionless, DimensionalityError, id='dimensionless'), pytest.param(unit_registry.s, DimensionalityError, id='incompatible_unit'), pytest.param(unit_registry.cm, None, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')))
    def test_numpy_methods(self, func, unit, error, dtype):
        array = np.linspace(0, 1, 10).astype(dtype) * unit_registry.m
        variable = xr.Variable('x', array)
        args = [item * unit if isinstance(item, (int, float, list)) else item for item in func.args]
        kwargs = {key: value * unit if isinstance(value, (int, float, list)) else value for key, value in func.kwargs.items()}
        if error is not None and func.name in ('searchsorted', 'clip'):
            with pytest.raises(error):
                func(variable, *args, **kwargs)
            return
        converted_args = [strip_units(convert_units(item, {None: unit_registry.m})) for item in args]
        converted_kwargs = {key: strip_units(convert_units(value, {None: unit_registry.m})) for key, value in kwargs.items()}
        units = extract_units(func(array, *args, **kwargs))
        expected = attach_units(func(strip_units(variable), *converted_args, **converted_kwargs), units)
        actual = func(variable, *args, **kwargs)
        assert_units_equal(expected, actual)
        assert_allclose(expected, actual)

    @pytest.mark.parametrize('func', (method('item', 5), method('searchsorted', 5)), ids=repr)
    @pytest.mark.parametrize('unit,error', (pytest.param(1, DimensionalityError, id='no_unit'), pytest.param(unit_registry.dimensionless, DimensionalityError, id='dimensionless'), pytest.param(unit_registry.s, DimensionalityError, id='incompatible_unit'), pytest.param(unit_registry.cm, None, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')))
    def test_raw_numpy_methods(self, func, unit, error, dtype):
        array = np.linspace(0, 1, 10).astype(dtype) * unit_registry.m
        variable = xr.Variable('x', array)
        args = [item * unit if isinstance(item, (int, float, list)) and func.name != 'item' else item for item in func.args]
        kwargs = {key: value * unit if isinstance(value, (int, float, list)) and func.name != 'item' else value for key, value in func.kwargs.items()}
        if error is not None and func.name != 'item':
            with pytest.raises(error):
                func(variable, *args, **kwargs)
            return
        converted_args = [strip_units(convert_units(item, {None: unit_registry.m})) if func.name != 'item' else item for item in args]
        converted_kwargs = {key: strip_units(convert_units(value, {None: unit_registry.m})) if func.name != 'item' else value for key, value in kwargs.items()}
        units = extract_units(func(array, *args, **kwargs))
        expected = attach_units(func(strip_units(variable), *converted_args, **converted_kwargs), units)
        actual = func(variable, *args, **kwargs)
        assert_units_equal(expected, actual)
        assert_duckarray_allclose(expected, actual)

    @pytest.mark.parametrize('func', (method('isnull'), method('notnull'), method('count')), ids=repr)
    def test_missing_value_detection(self, func):
        array = np.array([[1.4, 2.3, np.nan, 7.2], [np.nan, 9.7, np.nan, np.nan], [2.1, np.nan, np.nan, 4.6], [9.9, np.nan, 7.2, 9.1]]) * unit_registry.degK
        variable = xr.Variable(('x', 'y'), array)
        expected = func(strip_units(variable))
        actual = func(variable)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('unit,error', (pytest.param(1, DimensionalityError, id='no_unit'), pytest.param(unit_registry.dimensionless, DimensionalityError, id='dimensionless'), pytest.param(unit_registry.s, DimensionalityError, id='incompatible_unit'), pytest.param(unit_registry.cm, None, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')))
    def test_missing_value_fillna(self, unit, error):
        value = 10
        array = np.array([[1.4, 2.3, np.nan, 7.2], [np.nan, 9.7, np.nan, np.nan], [2.1, np.nan, np.nan, 4.6], [9.9, np.nan, 7.2, 9.1]]) * unit_registry.m
        variable = xr.Variable(('x', 'y'), array)
        fill_value = value * unit
        if error is not None:
            with pytest.raises(error):
                variable.fillna(value=fill_value)
            return
        expected = attach_units(strip_units(variable).fillna(value=fill_value.to(unit_registry.m).magnitude), extract_units(variable))
        actual = variable.fillna(value=fill_value)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('unit', (pytest.param(1, id='no_unit'), pytest.param(unit_registry.dimensionless, id='dimensionless'), pytest.param(unit_registry.s, id='incompatible_unit'), pytest.param(unit_registry.cm, id='compatible_unit'), pytest.param(unit_registry.m, id='identical_unit')))
    @pytest.mark.parametrize('convert_data', (pytest.param(False, id='no_conversion'), pytest.param(True, id='with_conversion')))
    @pytest.mark.parametrize('func', (method('equals'), pytest.param(method('identical'), marks=pytest.mark.skip(reason='behavior of identical is undecided'))), ids=repr)
    def test_comparisons(self, func, unit, convert_data, dtype):
        array = np.linspace(0, 1, 9).astype(dtype)
        quantity1 = array * unit_registry.m
        variable = xr.Variable('x', quantity1)
        if convert_data and is_compatible(unit_registry.m, unit):
            quantity2 = convert_units(array * unit_registry.m, {None: unit})
        else:
            quantity2 = array * unit
        other = xr.Variable('x', quantity2)
        expected = func(strip_units(variable), strip_units(convert_units(other, extract_units(variable)) if is_compatible(unit_registry.m, unit) else other))
        if func.name == 'identical':
            expected &= extract_units(variable) == extract_units(other)
        else:
            expected &= all(compatible_mappings(extract_units(variable), extract_units(other)).values())
        actual = func(variable, other)
        assert expected == actual

    @pytest.mark.parametrize('unit', (pytest.param(1, id='no_unit'), pytest.param(unit_registry.dimensionless, id='dimensionless'), pytest.param(unit_registry.s, id='incompatible_unit'), pytest.param(unit_registry.cm, id='compatible_unit'), pytest.param(unit_registry.m, id='identical_unit')))
    def test_broadcast_equals(self, unit, dtype):
        base_unit = unit_registry.m
        left_array = np.ones(shape=(2, 2), dtype=dtype) * base_unit
        value = (1 * base_unit).to(unit).magnitude if is_compatible(unit, base_unit) else 1
        right_array = np.full(shape=(2,), fill_value=value, dtype=dtype) * unit
        left = xr.Variable(('x', 'y'), left_array)
        right = xr.Variable('x', right_array)
        units = {**extract_units(left), **({} if is_compatible(unit, base_unit) else {None: None})}
        expected = strip_units(left).broadcast_equals(strip_units(convert_units(right, units))) & is_compatible(unit, base_unit)
        actual = left.broadcast_equals(right)
        assert expected == actual

    @pytest.mark.parametrize('dask', [False, pytest.param(True, marks=[requires_dask])])
    @pytest.mark.parametrize(['variable', 'indexers'], (pytest.param(xr.Variable('x', np.linspace(0, 5, 10)), {'x': 4}, id='single value-single indexer'), pytest.param(xr.Variable('x', np.linspace(0, 5, 10)), {'x': [5, 2, 9, 1]}, id='multiple values-single indexer'), pytest.param(xr.Variable(('x', 'y'), np.linspace(0, 5, 20).reshape(4, 5)), {'x': 1, 'y': 4}, id='single value-multiple indexers'), pytest.param(xr.Variable(('x', 'y'), np.linspace(0, 5, 20).reshape(4, 5)), {'x': [0, 1, 2], 'y': [0, 2, 4]}, id='multiple values-multiple indexers')))
    def test_isel(self, variable, indexers, dask, dtype):
        if dask:
            variable = variable.chunk({dim: 2 for dim in variable.dims})
        quantified = xr.Variable(variable.dims, variable.data.astype(dtype) * unit_registry.s)
        expected = attach_units(strip_units(quantified).isel(indexers), extract_units(quantified))
        actual = quantified.isel(indexers)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('unit,error', (pytest.param(1, DimensionalityError, id='no_unit'), pytest.param(unit_registry.dimensionless, DimensionalityError, id='dimensionless'), pytest.param(unit_registry.s, DimensionalityError, id='incompatible_unit'), pytest.param(unit_registry.cm, None, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')))
    @pytest.mark.parametrize('func', (function(lambda x, *_: +x, function_label='unary_plus'), function(lambda x, *_: -x, function_label='unary_minus'), function(lambda x, *_: abs(x), function_label='absolute'), function(lambda x, y: x + y, function_label='sum'), function(lambda x, y: y + x, function_label='commutative_sum'), function(lambda x, y: x * y, function_label='product'), function(lambda x, y: y * x, function_label='commutative_product')), ids=repr)
    def test_1d_math(self, func, unit, error, dtype):
        base_unit = unit_registry.m
        array = np.arange(5).astype(dtype) * base_unit
        variable = xr.Variable('x', array)
        values = np.ones(5)
        y = values * unit
        if error is not None and func.name in ('sum', 'commutative_sum'):
            with pytest.raises(error):
                func(variable, y)
            return
        units = extract_units(func(array, y))
        if all(compatible_mappings(units, extract_units(y)).values()):
            converted_y = convert_units(y, units)
        else:
            converted_y = y
        if all(compatible_mappings(units, extract_units(variable)).values()):
            converted_variable = convert_units(variable, units)
        else:
            converted_variable = variable
        expected = attach_units(func(strip_units(converted_variable), strip_units(converted_y)), units)
        actual = func(variable, y)
        assert_units_equal(expected, actual)
        assert_allclose(expected, actual)

    @pytest.mark.parametrize('unit,error', (pytest.param(1, DimensionalityError, id='no_unit'), pytest.param(unit_registry.dimensionless, DimensionalityError, id='dimensionless'), pytest.param(unit_registry.s, DimensionalityError, id='incompatible_unit'), pytest.param(unit_registry.cm, None, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')))
    @pytest.mark.parametrize('func', (method('where'), method('_getitem_with_mask')), ids=repr)
    def test_masking(self, func, unit, error, dtype):
        base_unit = unit_registry.m
        array = np.linspace(0, 5, 10).astype(dtype) * base_unit
        variable = xr.Variable('x', array)
        cond = np.array([True, False] * 5)
        other = -1 * unit
        if error is not None:
            with pytest.raises(error):
                func(variable, cond, other)
            return
        expected = attach_units(func(strip_units(variable), cond, strip_units(convert_units(other, {None: base_unit} if is_compatible(base_unit, unit) else {None: None}))), extract_units(variable))
        actual = func(variable, cond, other)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('dim', ('x', 'y', 'z', 't', 'all'))
    def test_squeeze(self, dim, dtype):
        shape = (2, 1, 3, 1, 1, 2)
        names = list('abcdef')
        dim_lengths = dict(zip(names, shape))
        array = np.ones(shape=shape) * unit_registry.m
        variable = xr.Variable(names, array)
        kwargs = {'dim': dim} if dim != 'all' and dim_lengths.get(dim, 0) == 1 else {}
        expected = attach_units(strip_units(variable).squeeze(**kwargs), extract_units(variable))
        actual = variable.squeeze(**kwargs)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('compute_backend', ['numbagg', None], indirect=True)
    @pytest.mark.parametrize('func', (method('coarsen', windows={'y': 2}, func=np.mean), method('quantile', q=[0.25, 0.75]), pytest.param(method('rank', dim='x'), marks=pytest.mark.skip(reason='rank not implemented for non-ndarray')), method('roll', {'x': 2}), pytest.param(method('rolling_window', 'x', 3, 'window'), marks=pytest.mark.xfail(reason='converts to ndarray')), method('reduce', np.std, 'x'), method('round', 2), method('shift', {'x': -2}), method('transpose', 'y', 'x')), ids=repr)
    def test_computation(self, func, dtype, compute_backend):
        base_unit = unit_registry.m
        array = np.linspace(0, 5, 5 * 10).reshape(5, 10).astype(dtype) * base_unit
        variable = xr.Variable(('x', 'y'), array)
        expected = attach_units(func(strip_units(variable)), extract_units(variable))
        actual = func(variable)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('unit,error', (pytest.param(1, DimensionalityError, id='no_unit'), pytest.param(unit_registry.dimensionless, DimensionalityError, id='dimensionless'), pytest.param(unit_registry.s, DimensionalityError, id='incompatible_unit'), pytest.param(unit_registry.cm, None, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')))
    def test_searchsorted(self, unit, error, dtype):
        base_unit = unit_registry.m
        array = np.linspace(0, 5, 10).astype(dtype) * base_unit
        variable = xr.Variable('x', array)
        value = 0 * unit
        if error is not None:
            with pytest.raises(error):
                variable.searchsorted(value)
            return
        expected = strip_units(variable).searchsorted(strip_units(convert_units(value, {None: base_unit})))
        actual = variable.searchsorted(value)
        assert_units_equal(expected, actual)
        np.testing.assert_allclose(expected, actual)

    def test_stack(self, dtype):
        array = np.linspace(0, 5, 3 * 10).reshape(3, 10).astype(dtype) * unit_registry.m
        variable = xr.Variable(('x', 'y'), array)
        expected = attach_units(strip_units(variable).stack(z=('x', 'y')), extract_units(variable))
        actual = variable.stack(z=('x', 'y'))
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    def test_unstack(self, dtype):
        array = np.linspace(0, 5, 3 * 10).astype(dtype) * unit_registry.m
        variable = xr.Variable('z', array)
        expected = attach_units(strip_units(variable).unstack(z={'x': 3, 'y': 10}), extract_units(variable))
        actual = variable.unstack(z={'x': 3, 'y': 10})
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('unit,error', (pytest.param(1, DimensionalityError, id='no_unit'), pytest.param(unit_registry.dimensionless, DimensionalityError, id='dimensionless'), pytest.param(unit_registry.s, DimensionalityError, id='incompatible_unit'), pytest.param(unit_registry.cm, None, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')))
    def test_concat(self, unit, error, dtype):
        array1 = np.linspace(0, 5, 9 * 10).reshape(3, 6, 5).astype(dtype) * unit_registry.m
        array2 = np.linspace(5, 10, 10 * 3).reshape(3, 2, 5).astype(dtype) * unit
        variable = xr.Variable(('x', 'y', 'z'), array1)
        other = xr.Variable(('x', 'y', 'z'), array2)
        if error is not None:
            with pytest.raises(error):
                xr.Variable.concat([variable, other], dim='y')
            return
        units = extract_units(variable)
        expected = attach_units(xr.Variable.concat([strip_units(variable), strip_units(convert_units(other, units))], dim='y'), units)
        actual = xr.Variable.concat([variable, other], dim='y')
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    def test_set_dims(self, dtype):
        array = np.linspace(0, 5, 3 * 10).reshape(3, 10).astype(dtype) * unit_registry.m
        variable = xr.Variable(('x', 'y'), array)
        dims = {'z': 6, 'x': 3, 'a': 1, 'b': 4, 'y': 10}
        expected = attach_units(strip_units(variable).set_dims(dims), extract_units(variable))
        actual = variable.set_dims(dims)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    def test_copy(self, dtype):
        array = np.linspace(0, 5, 10).astype(dtype) * unit_registry.m
        other = np.arange(10).astype(dtype) * unit_registry.s
        variable = xr.Variable('x', array)
        expected = attach_units(strip_units(variable).copy(data=strip_units(other)), extract_units(other))
        actual = variable.copy(data=other)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)

    @pytest.mark.parametrize('unit', (pytest.param(1, id='no_unit'), pytest.param(unit_registry.dimensionless, id='dimensionless'), pytest.param(unit_registry.s, id='incompatible_unit'), pytest.param(unit_registry.cm, id='compatible_unit'), pytest.param(unit_registry.m, id='identical_unit')))
    def test_no_conflicts(self, unit, dtype):
        base_unit = unit_registry.m
        array1 = np.array([[6.3, 0.3, 0.45], [np.nan, 0.3, 0.3], [3.7, np.nan, 0.2], [9.43, 0.3, 0.7]]) * base_unit
        array2 = np.array([np.nan, 0.3, np.nan]) * unit
        variable = xr.Variable(('x', 'y'), array1)
        other = xr.Variable('y', array2)
        expected = strip_units(variable).no_conflicts(strip_units(convert_units(other, {None: base_unit if is_compatible(base_unit, unit) else None}))) & is_compatible(base_unit, unit)
        actual = variable.no_conflicts(other)
        assert expected == actual

    @pytest.mark.parametrize('mode', ['constant', 'mean', 'median', 'reflect', 'edge', 'linear_ramp', 'maximum', 'minimum', 'symmetric', 'wrap'])
    @pytest.mark.parametrize('xr_arg, np_arg', _PAD_XR_NP_ARGS)
    def test_pad(self, mode, xr_arg, np_arg):
        data = np.arange(4 * 3 * 2).reshape(4, 3, 2) * unit_registry.m
        v = xr.Variable(['x', 'y', 'z'], data)
        expected = attach_units(strip_units(v).pad(mode=mode, **xr_arg), extract_units(v))
        actual = v.pad(mode=mode, **xr_arg)
        assert_units_equal(expected, actual)
        assert_equal(actual, expected)

    @pytest.mark.parametrize('unit,error', (pytest.param(1, DimensionalityError, id='no_unit'), pytest.param(unit_registry.dimensionless, DimensionalityError, id='dimensionless'), pytest.param(unit_registry.s, DimensionalityError, id='incompatible_unit'), pytest.param(unit_registry.cm, None, id='compatible_unit'), pytest.param(unit_registry.m, None, id='identical_unit')))
    def test_pad_unit_constant_value(self, unit, error, dtype):
        array = np.linspace(0, 5, 3 * 10).reshape(3, 10).astype(dtype) * unit_registry.m
        variable = xr.Variable(('x', 'y'), array)
        fill_value = -100 * unit
        func = method('pad', mode='constant', x=(2, 3), y=(1, 4))
        if error is not None:
            with pytest.raises(error):
                func(variable, constant_values=fill_value)
            return
        units = extract_units(variable)
        expected = attach_units(func(strip_units(variable), constant_values=strip_units(convert_units(fill_value, units))), units)
        actual = func(variable, constant_values=fill_value)
        assert_units_equal(expected, actual)
        assert_identical(expected, actual)