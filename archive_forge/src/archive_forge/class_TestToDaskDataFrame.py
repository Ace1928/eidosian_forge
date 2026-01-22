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
class TestToDaskDataFrame:

    def test_to_dask_dataframe(self):
        x = np.random.randn(10)
        y = np.arange(10, dtype='uint8')
        t = list('abcdefghij')
        ds = Dataset({'a': ('t', da.from_array(x, chunks=4)), 'b': ('t', y), 't': ('t', t)})
        expected_pd = pd.DataFrame({'a': x, 'b': y}, index=pd.Index(t, name='t'))
        expected = dd.from_pandas(expected_pd, chunksize=4)
        actual = ds.to_dask_dataframe(set_index=True)
        assert isinstance(actual, dd.DataFrame)
        assert_frame_equal(actual.compute(), expected.compute())
        expected = dd.from_pandas(expected_pd.reset_index(drop=False), chunksize=4)
        actual = ds.to_dask_dataframe(set_index=False)
        assert isinstance(actual, dd.DataFrame)
        assert_frame_equal(actual.compute(), expected.compute())

    @pytest.mark.xfail(reason='Currently pandas with pyarrow installed will return a `string[pyarrow]` type, which causes the `y` column to have a different type depending on whether pyarrow is installed')
    def test_to_dask_dataframe_2D(self):
        w = np.random.randn(2, 3)
        ds = Dataset({'w': (('x', 'y'), da.from_array(w, chunks=(1, 2)))})
        ds['x'] = ('x', np.array([0, 1], np.int64))
        ds['y'] = ('y', list('abc'))
        exp_index = pd.MultiIndex.from_arrays([[0, 0, 0, 1, 1, 1], ['a', 'b', 'c', 'a', 'b', 'c']], names=['x', 'y'])
        expected = pd.DataFrame({'w': w.reshape(-1)}, index=exp_index)
        expected = expected.reset_index(drop=False)
        actual = ds.to_dask_dataframe(set_index=False)
        assert isinstance(actual, dd.DataFrame)
        assert_frame_equal(actual.compute(), expected)

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_to_dask_dataframe_2D_set_index(self):
        w = da.from_array(np.random.randn(2, 3), chunks=(1, 2))
        ds = Dataset({'w': (('x', 'y'), w)})
        ds['x'] = ('x', np.array([0, 1], np.int64))
        ds['y'] = ('y', list('abc'))
        expected = ds.compute().to_dataframe()
        actual = ds.to_dask_dataframe(set_index=True)
        assert isinstance(actual, dd.DataFrame)
        assert_frame_equal(expected, actual.compute())

    def test_to_dask_dataframe_coordinates(self):
        x = np.random.randn(10)
        t = np.arange(10) * 2
        ds = Dataset({'a': ('t', da.from_array(x, chunks=4)), 't': ('t', da.from_array(t, chunks=4))})
        expected_pd = pd.DataFrame({'a': x}, index=pd.Index(t, name='t'))
        expected = dd.from_pandas(expected_pd, chunksize=4)
        actual = ds.to_dask_dataframe(set_index=True)
        assert isinstance(actual, dd.DataFrame)
        assert_frame_equal(expected.compute(), actual.compute())

    @pytest.mark.xfail(reason='Currently pandas with pyarrow installed will return a `string[pyarrow]` type, which causes the index to have a different type depending on whether pyarrow is installed')
    def test_to_dask_dataframe_not_daskarray(self):
        x = np.random.randn(10)
        y = np.arange(10, dtype='uint8')
        t = list('abcdefghij')
        ds = Dataset({'a': ('t', x), 'b': ('t', y), 't': ('t', t)})
        expected = pd.DataFrame({'a': x, 'b': y}, index=pd.Index(t, name='t'))
        actual = ds.to_dask_dataframe(set_index=True)
        assert isinstance(actual, dd.DataFrame)
        assert_frame_equal(expected, actual.compute())

    def test_to_dask_dataframe_no_coordinate(self):
        x = da.from_array(np.random.randn(10), chunks=4)
        ds = Dataset({'x': ('dim_0', x)})
        expected = ds.compute().to_dataframe().reset_index()
        actual = ds.to_dask_dataframe()
        assert isinstance(actual, dd.DataFrame)
        assert_frame_equal(expected, actual.compute())
        expected = ds.compute().to_dataframe()
        actual = ds.to_dask_dataframe(set_index=True)
        assert isinstance(actual, dd.DataFrame)
        assert_frame_equal(expected, actual.compute())

    def test_to_dask_dataframe_dim_order(self):
        values = np.array([[1, 2], [3, 4]], dtype=np.int64)
        ds = Dataset({'w': (('x', 'y'), values)}).chunk(1)
        expected = ds['w'].to_series().reset_index()
        actual = ds.to_dask_dataframe(dim_order=['x', 'y'])
        assert isinstance(actual, dd.DataFrame)
        assert_frame_equal(expected, actual.compute())
        expected = ds['w'].T.to_series().reset_index()
        actual = ds.to_dask_dataframe(dim_order=['y', 'x'])
        assert isinstance(actual, dd.DataFrame)
        assert_frame_equal(expected, actual.compute())
        with pytest.raises(ValueError, match='does not match the set of dimensions'):
            ds.to_dask_dataframe(dim_order=['x'])