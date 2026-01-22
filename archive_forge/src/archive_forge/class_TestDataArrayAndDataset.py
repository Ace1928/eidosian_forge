from __future__ import annotations
import operator
import pickle
import sys
from contextlib import suppress
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, Variable
from xarray.core import duck_array_ops
from xarray.core.duck_array_ops import lazy_array_equiv
from xarray.testing import assert_chunks_equal
from xarray.tests import (
from xarray.tests.test_backends import create_tmp_file
class TestDataArrayAndDataset(DaskTestCase):

    def assertLazyAndIdentical(self, expected, actual):
        self.assertLazyAnd(expected, actual, assert_identical)

    def assertLazyAndAllClose(self, expected, actual):
        self.assertLazyAnd(expected, actual, assert_allclose)

    def assertLazyAndEqual(self, expected, actual):
        self.assertLazyAnd(expected, actual, assert_equal)

    @pytest.fixture(autouse=True)
    def setUp(self):
        self.values = np.random.randn(4, 6)
        self.data = da.from_array(self.values, chunks=(2, 2))
        self.eager_array = DataArray(self.values, coords={'x': range(4)}, dims=('x', 'y'), name='foo')
        self.lazy_array = DataArray(self.data, coords={'x': range(4)}, dims=('x', 'y'), name='foo')

    def test_chunk(self):
        for chunks, expected in [({}, ((2, 2), (2, 2, 2))), (3, ((3, 1), (3, 3))), ({'x': 3, 'y': 3}, ((3, 1), (3, 3))), ({'x': 3}, ((3, 1), (2, 2, 2))), ({'x': (3, 1)}, ((3, 1), (2, 2, 2)))]:
            rechunked = self.lazy_array.chunk(chunks)
            assert rechunked.chunks == expected
            self.assertLazyAndIdentical(self.eager_array, rechunked)
            expected_chunksizes = {dim: chunks for dim, chunks in zip(self.lazy_array.dims, expected)}
            assert rechunked.chunksizes == expected_chunksizes
            lazy_dataset = self.lazy_array.to_dataset()
            eager_dataset = self.eager_array.to_dataset()
            expected_chunksizes = {dim: chunks for dim, chunks in zip(lazy_dataset.dims, expected)}
            rechunked = lazy_dataset.chunk(chunks)
            assert rechunked.chunks == expected_chunksizes
            self.assertLazyAndIdentical(eager_dataset, rechunked)
            assert rechunked.chunksizes == expected_chunksizes

    def test_rechunk(self):
        chunked = self.eager_array.chunk({'x': 2}).chunk({'y': 2})
        assert chunked.chunks == ((2,) * 2, (2,) * 3)
        self.assertLazyAndIdentical(self.lazy_array, chunked)

    def test_new_chunk(self):
        chunked = self.eager_array.chunk()
        assert chunked.data.name.startswith('xarray-<this-array>')

    def test_lazy_dataset(self):
        lazy_ds = Dataset({'foo': (('x', 'y'), self.data)})
        assert isinstance(lazy_ds.foo.variable.data, da.Array)

    def test_lazy_array(self):
        u = self.eager_array
        v = self.lazy_array
        self.assertLazyAndAllClose(u, v)
        self.assertLazyAndAllClose(-u, -v)
        self.assertLazyAndAllClose(u.T, v.T)
        self.assertLazyAndAllClose(u.mean(), v.mean())
        self.assertLazyAndAllClose(1 + u, 1 + v)
        actual = xr.concat([v[:2], v[2:]], 'x')
        self.assertLazyAndAllClose(u, actual)

    def test_compute(self):
        u = self.eager_array
        v = self.lazy_array
        assert dask.is_dask_collection(v)
        v2, = dask.compute(v + 1)
        assert not dask.is_dask_collection(v2)
        assert ((u + 1).data == v2.data).all()

    def test_persist(self):
        u = self.eager_array
        v = self.lazy_array + 1
        v2, = dask.persist(v)
        assert v is not v2
        assert len(v2.__dask_graph__()) < len(v.__dask_graph__())
        assert v2.__dask_keys__() == v.__dask_keys__()
        assert dask.is_dask_collection(v)
        assert dask.is_dask_collection(v2)
        self.assertLazyAndAllClose(u + 1, v)
        self.assertLazyAndAllClose(u + 1, v2)

    def test_concat_loads_variables(self):
        d1 = build_dask_array('d1')
        c1 = build_dask_array('c1')
        d2 = build_dask_array('d2')
        c2 = build_dask_array('c2')
        d3 = build_dask_array('d3')
        c3 = build_dask_array('c3')
        ds1 = Dataset(data_vars={'d': ('x', d1)}, coords={'c': ('x', c1)})
        ds2 = Dataset(data_vars={'d': ('x', d2)}, coords={'c': ('x', c2)})
        ds3 = Dataset(data_vars={'d': ('x', d3)}, coords={'c': ('x', c3)})
        assert kernel_call_count == 0
        out = xr.concat([ds1, ds2, ds3], dim='n', data_vars='different', coords='different')
        assert kernel_call_count == 6
        assert isinstance(out['d'].data, np.ndarray)
        assert isinstance(out['c'].data, np.ndarray)
        out = xr.concat([ds1, ds2, ds3], dim='n', data_vars='all', coords='all')
        assert kernel_call_count == 6
        assert isinstance(out['d'].data, dask.array.Array)
        assert isinstance(out['c'].data, dask.array.Array)
        out = xr.concat([ds1, ds2, ds3], dim='n', data_vars=['d'], coords=['c'])
        assert kernel_call_count == 6
        assert isinstance(out['d'].data, dask.array.Array)
        assert isinstance(out['c'].data, dask.array.Array)
        out = xr.concat([ds1, ds2, ds3], dim='n', data_vars=[], coords=[])
        assert kernel_call_count == 12
        assert isinstance(out['d'].data, np.ndarray)
        assert isinstance(out['c'].data, np.ndarray)
        out = xr.concat([ds1, ds2, ds3], dim='n', data_vars='different', coords='different', compat='identical')
        assert kernel_call_count == 18
        assert isinstance(out['d'].data, np.ndarray)
        assert isinstance(out['c'].data, np.ndarray)
        ds4 = Dataset(data_vars={'d': ('x', [2.0])}, coords={'c': ('x', [2.0])})
        out = xr.concat([ds1, ds2, ds4, ds3], dim='n', data_vars='different', coords='different')
        assert kernel_call_count == 22
        assert isinstance(out['d'].data, dask.array.Array)
        assert isinstance(out['c'].data, dask.array.Array)
        out.compute()
        assert kernel_call_count == 24
        assert ds1['d'].data is d1
        assert ds1['c'].data is c1
        assert ds2['d'].data is d2
        assert ds2['c'].data is c2
        assert ds3['d'].data is d3
        assert ds3['c'].data is c3
        out = xr.concat([ds1, ds1, ds1], dim='n', data_vars='different', coords='different')
        assert kernel_call_count == 24
        assert isinstance(out['d'].data, dask.array.Array)
        assert isinstance(out['c'].data, dask.array.Array)
        out = xr.concat([ds1, ds1, ds1], dim='n', data_vars=[], coords=[], compat='identical')
        assert kernel_call_count == 24
        assert isinstance(out['d'].data, dask.array.Array)
        assert isinstance(out['c'].data, dask.array.Array)
        out = xr.concat([ds1, ds2.compute(), ds3], dim='n', data_vars='all', coords='different', compat='identical')
        assert kernel_call_count == 28
        out = xr.concat([ds1, ds2.compute(), ds3], dim='n', data_vars='all', coords='all', compat='identical')
        assert kernel_call_count == 30
        assert ds1['d'].data is d1
        assert ds1['c'].data is c1
        assert ds2['d'].data is d2
        assert ds2['c'].data is c2
        assert ds3['d'].data is d3
        assert ds3['c'].data is c3

    def test_groupby(self):
        u = self.eager_array
        v = self.lazy_array
        expected = u.groupby('x').mean(...)
        with raise_if_dask_computes():
            actual = v.groupby('x').mean(...)
        self.assertLazyAndAllClose(expected, actual)

    def test_rolling(self):
        u = self.eager_array
        v = self.lazy_array
        expected = u.rolling(x=2).mean()
        with raise_if_dask_computes():
            actual = v.rolling(x=2).mean()
        self.assertLazyAndAllClose(expected, actual)

    @pytest.mark.parametrize('func', ['first', 'last'])
    def test_groupby_first_last(self, func):
        method = operator.methodcaller(func)
        u = self.eager_array
        v = self.lazy_array
        for coords in [u.coords, v.coords]:
            coords['ab'] = ('x', ['a', 'a', 'b', 'b'])
        expected = method(u.groupby('ab'))
        with raise_if_dask_computes():
            actual = method(v.groupby('ab'))
        self.assertLazyAndAllClose(expected, actual)
        with raise_if_dask_computes():
            actual = method(v.groupby('ab'))
        self.assertLazyAndAllClose(expected, actual)

    def test_reindex(self):
        u = self.eager_array.assign_coords(y=range(6))
        v = self.lazy_array.assign_coords(y=range(6))
        for kwargs in [{'x': [2, 3, 4]}, {'x': [1, 100, 2, 101, 3]}, {'x': [2.5, 3, 3.5], 'y': [2, 2.5, 3]}]:
            expected = u.reindex(**kwargs)
            actual = v.reindex(**kwargs)
            self.assertLazyAndAllClose(expected, actual)

    def test_to_dataset_roundtrip(self):
        u = self.eager_array
        v = self.lazy_array
        expected = u.assign_coords(x=u['x'])
        self.assertLazyAndEqual(expected, v.to_dataset('x').to_dataarray('x'))

    def test_merge(self):

        def duplicate_and_merge(array):
            return xr.merge([array, array.rename('bar')]).to_dataarray()
        expected = duplicate_and_merge(self.eager_array)
        actual = duplicate_and_merge(self.lazy_array)
        self.assertLazyAndEqual(expected, actual)

    def test_ufuncs(self):
        u = self.eager_array
        v = self.lazy_array
        self.assertLazyAndAllClose(np.sin(u), np.sin(v))

    def test_where_dispatching(self):
        a = np.arange(10)
        b = a > 3
        x = da.from_array(a, 5)
        y = da.from_array(b, 5)
        expected = DataArray(a).where(b)
        self.assertLazyAndEqual(expected, DataArray(a).where(y))
        self.assertLazyAndEqual(expected, DataArray(x).where(b))
        self.assertLazyAndEqual(expected, DataArray(x).where(y))

    def test_simultaneous_compute(self):
        ds = Dataset({'foo': ('x', range(5)), 'bar': ('x', range(5))}).chunk()
        count = [0]

        def counting_get(*args, **kwargs):
            count[0] += 1
            return dask.get(*args, **kwargs)
        ds.load(scheduler=counting_get)
        assert count[0] == 1

    def test_stack(self):
        data = da.random.normal(size=(2, 3, 4), chunks=(1, 3, 4))
        arr = DataArray(data, dims=('w', 'x', 'y'))
        stacked = arr.stack(z=('x', 'y'))
        z = pd.MultiIndex.from_product([np.arange(3), np.arange(4)], names=['x', 'y'])
        expected = DataArray(data.reshape(2, -1), {'z': z}, dims=['w', 'z'])
        assert stacked.data.chunks == expected.data.chunks
        self.assertLazyAndEqual(expected, stacked)

    def test_dot(self):
        eager = self.eager_array.dot(self.eager_array[0])
        lazy = self.lazy_array.dot(self.lazy_array[0])
        self.assertLazyAndAllClose(eager, lazy)

    def test_dataarray_repr(self):
        data = build_dask_array('data')
        nonindex_coord = build_dask_array('coord')
        a = DataArray(data, dims=['x'], coords={'y': ('x', nonindex_coord)})
        expected = dedent(f"            <xarray.DataArray 'data' (x: 1)> Size: 8B\n            {data!r}\n            Coordinates:\n                y        (x) int64 8B dask.array<chunksize=(1,), meta=np.ndarray>\n            Dimensions without coordinates: x")
        assert expected == repr(a)
        assert kernel_call_count == 0

    def test_dataset_repr(self):
        data = build_dask_array('data')
        nonindex_coord = build_dask_array('coord')
        ds = Dataset(data_vars={'a': ('x', data)}, coords={'y': ('x', nonindex_coord)})
        expected = dedent('            <xarray.Dataset> Size: 16B\n            Dimensions:  (x: 1)\n            Coordinates:\n                y        (x) int64 8B dask.array<chunksize=(1,), meta=np.ndarray>\n            Dimensions without coordinates: x\n            Data variables:\n                a        (x) int64 8B dask.array<chunksize=(1,), meta=np.ndarray>')
        assert expected == repr(ds)
        assert kernel_call_count == 0

    def test_dataarray_pickle(self):
        data = build_dask_array('data')
        nonindex_coord = build_dask_array('coord')
        a1 = DataArray(data, dims=['x'], coords={'y': ('x', nonindex_coord)})
        a1.compute()
        assert not a1._in_memory
        assert not a1.coords['y']._in_memory
        assert kernel_call_count == 2
        a2 = pickle.loads(pickle.dumps(a1))
        assert kernel_call_count == 2
        assert_identical(a1, a2)
        assert not a1._in_memory
        assert not a2._in_memory
        assert not a1.coords['y']._in_memory
        assert not a2.coords['y']._in_memory

    def test_dataset_pickle(self):
        data = build_dask_array('data')
        nonindex_coord = build_dask_array('coord')
        ds1 = Dataset(data_vars={'a': ('x', data)}, coords={'y': ('x', nonindex_coord)})
        ds1.compute()
        assert not ds1['a']._in_memory
        assert not ds1['y']._in_memory
        assert kernel_call_count == 2
        ds2 = pickle.loads(pickle.dumps(ds1))
        assert kernel_call_count == 2
        assert_identical(ds1, ds2)
        assert not ds1['a']._in_memory
        assert not ds2['a']._in_memory
        assert not ds1['y']._in_memory
        assert not ds2['y']._in_memory

    def test_dataarray_getattr(self):
        data = build_dask_array('data')
        nonindex_coord = build_dask_array('coord')
        a = DataArray(data, dims=['x'], coords={'y': ('x', nonindex_coord)})
        with suppress(AttributeError):
            getattr(a, 'NOTEXIST')
        assert kernel_call_count == 0

    def test_dataset_getattr(self):
        data = build_dask_array('data')
        nonindex_coord = build_dask_array('coord')
        ds = Dataset(data_vars={'a': ('x', data)}, coords={'y': ('x', nonindex_coord)})
        with suppress(AttributeError):
            getattr(ds, 'NOTEXIST')
        assert kernel_call_count == 0

    def test_values(self):
        a = DataArray([1, 2]).chunk()
        assert not a._in_memory
        assert a.values.tolist() == [1, 2]
        assert not a._in_memory

    def test_from_dask_variable(self):
        a = DataArray(self.lazy_array.variable, coords={'x': range(4)}, name='foo')
        self.assertLazyAndIdentical(self.lazy_array, a)

    @requires_pint
    def test_tokenize_duck_dask_array(self):
        import pint
        unit_registry = pint.UnitRegistry()
        q = unit_registry.Quantity(self.data, unit_registry.meter)
        data_array = xr.DataArray(data=q, coords={'x': range(4)}, dims=('x', 'y'), name='foo')
        token = dask.base.tokenize(data_array)
        post_op = data_array + 5 * unit_registry.meter
        assert dask.base.tokenize(data_array) != dask.base.tokenize(post_op)
        assert dask.base.tokenize(data_array) == token