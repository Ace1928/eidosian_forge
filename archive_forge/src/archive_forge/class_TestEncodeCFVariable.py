from __future__ import annotations
import contextlib
import warnings
import numpy as np
import pandas as pd
import pytest
from xarray import (
from xarray.backends.common import WritableCFDataStore
from xarray.backends.memory import InMemoryDataStore
from xarray.conventions import decode_cf
from xarray.testing import assert_identical
from xarray.tests import (
from xarray.tests.test_backends import CFEncodedBase
@requires_cftime
class TestEncodeCFVariable:

    def test_incompatible_attributes(self) -> None:
        invalid_vars = [Variable(['t'], pd.date_range('2000-01-01', periods=3), {'units': 'foobar'}), Variable(['t'], pd.to_timedelta(['1 day']), {'units': 'foobar'}), Variable(['t'], [0, 1, 2], {'add_offset': 0}, {'add_offset': 2}), Variable(['t'], [0, 1, 2], {'_FillValue': 0}, {'_FillValue': 2})]
        for var in invalid_vars:
            with pytest.raises(ValueError):
                conventions.encode_cf_variable(var)

    def test_missing_fillvalue(self) -> None:
        v = Variable(['x'], np.array([np.nan, 1, 2, 3]))
        v.encoding = {'dtype': 'int16'}
        with pytest.warns(Warning, match='floating point data as an integer'):
            conventions.encode_cf_variable(v)

    def test_multidimensional_coordinates(self) -> None:
        zeros1 = np.zeros((1, 5, 3))
        zeros2 = np.zeros((1, 6, 3))
        zeros3 = np.zeros((1, 5, 4))
        orig = Dataset({'lon1': (['x1', 'y1'], zeros1.squeeze(0), {}), 'lon2': (['x2', 'y1'], zeros2.squeeze(0), {}), 'lon3': (['x1', 'y2'], zeros3.squeeze(0), {}), 'lat1': (['x1', 'y1'], zeros1.squeeze(0), {}), 'lat2': (['x2', 'y1'], zeros2.squeeze(0), {}), 'lat3': (['x1', 'y2'], zeros3.squeeze(0), {}), 'foo1': (['time', 'x1', 'y1'], zeros1, {'coordinates': 'lon1 lat1'}), 'foo2': (['time', 'x2', 'y1'], zeros2, {'coordinates': 'lon2 lat2'}), 'foo3': (['time', 'x1', 'y2'], zeros3, {'coordinates': 'lon3 lat3'}), 'time': ('time', [0.0], {'units': 'hours since 2017-01-01'})})
        orig = conventions.decode_cf(orig)
        enc, attrs = conventions.encode_dataset_coordinates(orig)
        foo1_coords = enc['foo1'].attrs.get('coordinates', '')
        foo2_coords = enc['foo2'].attrs.get('coordinates', '')
        foo3_coords = enc['foo3'].attrs.get('coordinates', '')
        assert foo1_coords == 'lon1 lat1'
        assert foo2_coords == 'lon2 lat2'
        assert foo3_coords == 'lon3 lat3'
        assert 'coordinates' not in attrs

    def test_var_with_coord_attr(self) -> None:
        orig = Dataset({'values': ('time', np.zeros(2), {'coordinates': 'time lon lat'})}, coords={'time': ('time', np.zeros(2)), 'lat': ('time', np.zeros(2)), 'lon': ('time', np.zeros(2))})
        enc, attrs = conventions.encode_dataset_coordinates(orig)
        values_coords = enc['values'].attrs.get('coordinates', '')
        assert values_coords == 'time lon lat'
        assert 'coordinates' not in attrs

    def test_do_not_overwrite_user_coordinates(self) -> None:
        orig = Dataset(coords={'x': [0, 1, 2], 'y': ('x', [5, 6, 7]), 'z': ('x', [8, 9, 10])}, data_vars={'a': ('x', [1, 2, 3]), 'b': ('x', [3, 5, 6])})
        orig['a'].encoding['coordinates'] = 'y'
        orig['b'].encoding['coordinates'] = 'z'
        enc, _ = conventions.encode_dataset_coordinates(orig)
        assert enc['a'].attrs['coordinates'] == 'y'
        assert enc['b'].attrs['coordinates'] == 'z'
        orig['a'].attrs['coordinates'] = 'foo'
        with pytest.raises(ValueError, match="'coordinates' found in both attrs"):
            conventions.encode_dataset_coordinates(orig)

    def test_deterministic_coords_encoding(self) -> None:
        ds = Dataset({'foo': 0}, coords={'baz': 0, 'bar': 0})
        vars, attrs = conventions.encode_dataset_coordinates(ds)
        assert vars['foo'].attrs['coordinates'] == 'bar baz'
        assert attrs.get('coordinates') is None
        ds = ds.drop_vars('foo')
        vars, attrs = conventions.encode_dataset_coordinates(ds)
        assert attrs['coordinates'] == 'bar baz'

    @pytest.mark.filterwarnings('ignore:Converting non-nanosecond')
    def test_emit_coordinates_attribute_in_attrs(self) -> None:
        orig = Dataset({'a': 1, 'b': 1}, coords={'t': np.array('2004-11-01T00:00:00', dtype=np.datetime64)})
        orig['a'].attrs['coordinates'] = None
        enc, _ = conventions.encode_dataset_coordinates(orig)
        assert 'coordinates' not in enc['a'].attrs
        assert 'coordinates' not in enc['a'].encoding
        assert enc['b'].attrs.get('coordinates') == 't'
        assert 'coordinates' not in enc['b'].encoding

    @pytest.mark.filterwarnings('ignore:Converting non-nanosecond')
    def test_emit_coordinates_attribute_in_encoding(self) -> None:
        orig = Dataset({'a': 1, 'b': 1}, coords={'t': np.array('2004-11-01T00:00:00', dtype=np.datetime64)})
        orig['a'].encoding['coordinates'] = None
        enc, _ = conventions.encode_dataset_coordinates(orig)
        assert 'coordinates' not in enc['a'].attrs
        assert 'coordinates' not in enc['a'].encoding
        assert enc['b'].attrs.get('coordinates') == 't'
        assert 'coordinates' not in enc['b'].encoding

    @requires_dask
    def test_string_object_warning(self) -> None:
        original = Variable(('x',), np.array(['foo', 'bar'], dtype=object)).chunk()
        with pytest.warns(SerializationWarning, match='dask array with dtype=object'):
            encoded = conventions.encode_cf_variable(original)
        assert_identical(original, encoded)