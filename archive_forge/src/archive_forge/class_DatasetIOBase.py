from __future__ import annotations
import contextlib
import gzip
import itertools
import math
import os.path
import pickle
import platform
import re
import shutil
import sys
import tempfile
import uuid
import warnings
from collections.abc import Generator, Iterator, Mapping
from contextlib import ExitStack
from io import BytesIO
from os import listdir
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, cast
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
from pandas.errors import OutOfBoundsDatetime
import xarray as xr
from xarray import (
from xarray.backends.common import robust_getitem
from xarray.backends.h5netcdf_ import H5netcdfBackendEntrypoint
from xarray.backends.netcdf3 import _nc3_dtype_coercions
from xarray.backends.netCDF4_ import (
from xarray.backends.pydap_ import PydapDataStore
from xarray.backends.scipy_ import ScipyBackendEntrypoint
from xarray.coding.cftime_offsets import cftime_range
from xarray.coding.strings import check_vlen_dtype, create_vlen_dtype
from xarray.coding.variables import SerializationWarning
from xarray.conventions import encode_dataset_coordinates
from xarray.core import indexing
from xarray.core.options import set_options
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
from xarray.tests.test_coding_times import (
from xarray.tests.test_dataset import (
class DatasetIOBase:
    engine: T_NetcdfEngine | None = None
    file_format: T_NetcdfTypes | None = None

    def create_store(self):
        raise NotImplementedError()

    @contextlib.contextmanager
    def roundtrip(self, data, save_kwargs=None, open_kwargs=None, allow_cleanup_failure=False):
        if save_kwargs is None:
            save_kwargs = {}
        if open_kwargs is None:
            open_kwargs = {}
        with create_tmp_file(allow_cleanup_failure=allow_cleanup_failure) as path:
            self.save(data, path, **save_kwargs)
            with self.open(path, **open_kwargs) as ds:
                yield ds

    @contextlib.contextmanager
    def roundtrip_append(self, data, save_kwargs=None, open_kwargs=None, allow_cleanup_failure=False):
        if save_kwargs is None:
            save_kwargs = {}
        if open_kwargs is None:
            open_kwargs = {}
        with create_tmp_file(allow_cleanup_failure=allow_cleanup_failure) as path:
            for i, key in enumerate(data.variables):
                mode = 'a' if i > 0 else 'w'
                self.save(data[[key]], path, mode=mode, **save_kwargs)
            with self.open(path, **open_kwargs) as ds:
                yield ds

    def save(self, dataset, path, **kwargs):
        return dataset.to_netcdf(path, engine=self.engine, format=self.file_format, **kwargs)

    @contextlib.contextmanager
    def open(self, path, **kwargs):
        with open_dataset(path, engine=self.engine, **kwargs) as ds:
            yield ds

    def test_zero_dimensional_variable(self) -> None:
        expected = create_test_data()
        expected['float_var'] = ([], 1000000000.0, {'units': 'units of awesome'})
        expected['bytes_var'] = ([], b'foobar')
        expected['string_var'] = ([], 'foobar')
        with self.roundtrip(expected) as actual:
            assert_identical(expected, actual)

    def test_write_store(self) -> None:
        expected = create_test_data()
        with self.create_store() as store:
            expected.dump_to_store(store)
            with xr.decode_cf(store) as actual:
                assert_allclose(expected, actual)

    def check_dtypes_roundtripped(self, expected, actual):
        for k in expected.variables:
            expected_dtype = expected.variables[k].dtype
            if isinstance(self, NetCDF3Only) and str(expected_dtype) in _nc3_dtype_coercions:
                expected_dtype = np.dtype(_nc3_dtype_coercions[str(expected_dtype)])
            actual_dtype = actual.variables[k].dtype
            string_kinds = {'O', 'S', 'U'}
            assert expected_dtype == actual_dtype or (expected_dtype.kind in string_kinds and actual_dtype.kind in string_kinds)

    def test_roundtrip_test_data(self) -> None:
        expected = create_test_data()
        with self.roundtrip(expected) as actual:
            self.check_dtypes_roundtripped(expected, actual)
            assert_identical(expected, actual)

    def test_load(self) -> None:
        expected = create_test_data()

        @contextlib.contextmanager
        def assert_loads(vars=None):
            if vars is None:
                vars = expected
            with self.roundtrip(expected) as actual:
                for k, v in actual.variables.items():
                    assert v._in_memory == (k in actual.dims)
                yield actual
                for k, v in actual.variables.items():
                    if k in vars:
                        assert v._in_memory
                assert_identical(expected, actual)
        with pytest.raises(AssertionError):
            with assert_loads() as ds:
                pass
        with assert_loads() as ds:
            ds.load()
        with assert_loads(['var1', 'dim1', 'dim2']) as ds:
            ds['var1'].load()
        with self.roundtrip(expected) as ds:
            actual = ds.load()
        assert_identical(expected, actual)

    def test_dataset_compute(self) -> None:
        expected = create_test_data()
        with self.roundtrip(expected) as actual:
            for k, v in actual.variables.items():
                assert v._in_memory == (k in actual.dims)
            computed = actual.compute()
            for k, v in actual.variables.items():
                assert v._in_memory == (k in actual.dims)
            for v in computed.variables.values():
                assert v._in_memory
            assert_identical(expected, actual)
            assert_identical(expected, computed)

    def test_pickle(self) -> None:
        expected = Dataset({'foo': ('x', [42])})
        with self.roundtrip(expected, allow_cleanup_failure=ON_WINDOWS) as roundtripped:
            with roundtripped:
                raw_pickle = pickle.dumps(roundtripped)
            with pickle.loads(raw_pickle) as unpickled_ds:
                assert_identical(expected, unpickled_ds)

    @pytest.mark.filterwarnings('ignore:deallocating CachingFileManager')
    def test_pickle_dataarray(self) -> None:
        expected = Dataset({'foo': ('x', [42])})
        with self.roundtrip(expected, allow_cleanup_failure=ON_WINDOWS) as roundtripped:
            with roundtripped:
                raw_pickle = pickle.dumps(roundtripped['foo'])
            unpickled = pickle.loads(raw_pickle)
            assert_identical(expected['foo'], unpickled)

    def test_dataset_caching(self) -> None:
        expected = Dataset({'foo': ('x', [5, 6, 7])})
        with self.roundtrip(expected) as actual:
            assert isinstance(actual.foo.variable._data, indexing.MemoryCachedArray)
            assert not actual.foo.variable._in_memory
            actual.foo.values
            assert actual.foo.variable._in_memory
        with self.roundtrip(expected, open_kwargs={'cache': False}) as actual:
            assert isinstance(actual.foo.variable._data, indexing.CopyOnWriteArray)
            assert not actual.foo.variable._in_memory
            actual.foo.values
            assert not actual.foo.variable._in_memory

    @pytest.mark.filterwarnings('ignore:deallocating CachingFileManager')
    def test_roundtrip_None_variable(self) -> None:
        expected = Dataset({None: (('x', 'y'), [[0, 1], [2, 3]])})
        with self.roundtrip(expected) as actual:
            assert_identical(expected, actual)

    def test_roundtrip_object_dtype(self) -> None:
        floats = np.array([0.0, 0.0, 1.0, 2.0, 3.0], dtype=object)
        floats_nans = np.array([np.nan, np.nan, 1.0, 2.0, 3.0], dtype=object)
        bytes_ = np.array([b'ab', b'cdef', b'g'], dtype=object)
        bytes_nans = np.array([b'ab', b'cdef', np.nan], dtype=object)
        strings = np.array(['ab', 'cdef', 'g'], dtype=object)
        strings_nans = np.array(['ab', 'cdef', np.nan], dtype=object)
        all_nans = np.array([np.nan, np.nan], dtype=object)
        original = Dataset({'floats': ('a', floats), 'floats_nans': ('a', floats_nans), 'bytes': ('b', bytes_), 'bytes_nans': ('b', bytes_nans), 'strings': ('b', strings), 'strings_nans': ('b', strings_nans), 'all_nans': ('c', all_nans), 'nan': ([], np.nan)})
        expected = original.copy(deep=True)
        with self.roundtrip(original) as actual:
            try:
                assert_identical(expected, actual)
            except AssertionError:
                expected['bytes_nans'][-1] = b''
                expected['strings_nans'][-1] = ''
                assert_identical(expected, actual)

    def test_roundtrip_string_data(self) -> None:
        expected = Dataset({'x': ('t', ['ab', 'cdef'])})
        with self.roundtrip(expected) as actual:
            assert_identical(expected, actual)

    def test_roundtrip_string_encoded_characters(self) -> None:
        expected = Dataset({'x': ('t', ['ab', 'cdef'])})
        expected['x'].encoding['dtype'] = 'S1'
        with self.roundtrip(expected) as actual:
            assert_identical(expected, actual)
            assert actual['x'].encoding['_Encoding'] == 'utf-8'
        expected['x'].encoding['_Encoding'] = 'ascii'
        with self.roundtrip(expected) as actual:
            assert_identical(expected, actual)
            assert actual['x'].encoding['_Encoding'] == 'ascii'

    def test_roundtrip_numpy_datetime_data(self) -> None:
        times = pd.to_datetime(['2000-01-01', '2000-01-02', 'NaT'])
        expected = Dataset({'t': ('t', times), 't0': times[0]})
        kwargs = {'encoding': {'t0': {'units': 'days since 1950-01-01'}}}
        with self.roundtrip(expected, save_kwargs=kwargs) as actual:
            assert_identical(expected, actual)
            assert actual.t0.encoding['units'] == 'days since 1950-01-01'

    @requires_cftime
    def test_roundtrip_cftime_datetime_data(self) -> None:
        from xarray.tests.test_coding_times import _all_cftime_date_types
        date_types = _all_cftime_date_types()
        for date_type in date_types.values():
            times = [date_type(1, 1, 1), date_type(1, 1, 2)]
            expected = Dataset({'t': ('t', times), 't0': times[0]})
            kwargs = {'encoding': {'t0': {'units': 'days since 0001-01-01'}}}
            expected_decoded_t = np.array(times)
            expected_decoded_t0 = np.array([date_type(1, 1, 1)])
            expected_calendar = times[0].calendar
            with warnings.catch_warnings():
                if expected_calendar in {'proleptic_gregorian', 'standard'}:
                    warnings.filterwarnings('ignore', 'Unable to decode time axis')
                with self.roundtrip(expected, save_kwargs=kwargs) as actual:
                    abs_diff = abs(actual.t.values - expected_decoded_t)
                    assert (abs_diff <= np.timedelta64(1, 's')).all()
                    assert actual.t.encoding['units'] == 'days since 0001-01-01 00:00:00.000000'
                    assert actual.t.encoding['calendar'] == expected_calendar
                    abs_diff = abs(actual.t0.values - expected_decoded_t0)
                    assert (abs_diff <= np.timedelta64(1, 's')).all()
                    assert actual.t0.encoding['units'] == 'days since 0001-01-01'
                    assert actual.t.encoding['calendar'] == expected_calendar

    def test_roundtrip_timedelta_data(self) -> None:
        time_deltas = pd.to_timedelta(['1h', '2h', 'NaT'])
        expected = Dataset({'td': ('td', time_deltas), 'td0': time_deltas[0]})
        with self.roundtrip(expected) as actual:
            assert_identical(expected, actual)

    def test_roundtrip_float64_data(self) -> None:
        expected = Dataset({'x': ('y', np.array([1.0, 2.0, np.pi], dtype='float64'))})
        with self.roundtrip(expected) as actual:
            assert_identical(expected, actual)

    def test_roundtrip_example_1_netcdf(self) -> None:
        with open_example_dataset('example_1.nc') as expected:
            with self.roundtrip(expected) as actual:
                assert_equal(expected, actual)

    def test_roundtrip_coordinates(self) -> None:
        original = Dataset({'foo': ('x', [0, 1])}, {'x': [2, 3], 'y': ('a', [42]), 'z': ('x', [4, 5])})
        with self.roundtrip(original) as actual:
            assert_identical(original, actual)
        original['foo'].encoding['coordinates'] = 'y'
        with self.roundtrip(original, open_kwargs={'decode_coords': False}) as expected:
            with self.roundtrip(expected, open_kwargs={'decode_coords': False}) as actual:
                assert_identical(expected, actual)

    def test_roundtrip_global_coordinates(self) -> None:
        original = Dataset({'foo': ('x', [0, 1])}, {'x': [2, 3], 'y': ('a', [42]), 'z': ('x', [4, 5])})
        with self.roundtrip(original) as actual:
            assert_identical(original, actual)
        _, attrs = encode_dataset_coordinates(original)
        assert attrs['coordinates'] == 'y'
        original.attrs['coordinates'] = 'foo'
        with pytest.warns(SerializationWarning):
            _, attrs = encode_dataset_coordinates(original)
            assert attrs['coordinates'] == 'foo'

    def test_roundtrip_coordinates_with_space(self) -> None:
        original = Dataset(coords={'x': 0, 'y z': 1})
        expected = Dataset({'y z': 1}, {'x': 0})
        with pytest.warns(SerializationWarning):
            with self.roundtrip(original) as actual:
                assert_identical(expected, actual)

    def test_roundtrip_boolean_dtype(self) -> None:
        original = create_boolean_data()
        assert original['x'].dtype == 'bool'
        with self.roundtrip(original) as actual:
            assert_identical(original, actual)
            assert actual['x'].dtype == 'bool'
            with self.roundtrip(actual) as actual2:
                assert_identical(original, actual2)
                assert actual2['x'].dtype == 'bool'

    def test_orthogonal_indexing(self) -> None:
        in_memory = create_test_data()
        with self.roundtrip(in_memory) as on_disk:
            indexers = {'dim1': [1, 2, 0], 'dim2': [3, 2, 0, 3], 'dim3': np.arange(5)}
            expected = in_memory.isel(indexers)
            actual = on_disk.isel(**indexers)
            assert not actual['var1'].variable._in_memory
            assert_identical(expected, actual)
            actual = on_disk.isel(**indexers)
            assert_identical(expected, actual)

    def test_vectorized_indexing(self) -> None:
        in_memory = create_test_data()
        with self.roundtrip(in_memory) as on_disk:
            indexers = {'dim1': DataArray([0, 2, 0], dims='a'), 'dim2': DataArray([0, 2, 3], dims='a')}
            expected = in_memory.isel(indexers)
            actual = on_disk.isel(**indexers)
            assert not actual['var1'].variable._in_memory
            assert_identical(expected, actual.load())
            actual = on_disk.isel(**indexers)
            assert_identical(expected, actual)

        def multiple_indexing(indexers):
            with self.roundtrip(in_memory) as on_disk:
                actual = on_disk['var3']
                expected = in_memory['var3']
                for ind in indexers:
                    actual = actual.isel(ind)
                    expected = expected.isel(ind)
                    assert not actual.variable._in_memory
                assert_identical(expected, actual.load())
        indexers2 = [{'dim1': DataArray([[0, 7], [2, 6], [3, 5]], dims=['a', 'b']), 'dim3': DataArray([[0, 4], [1, 3], [2, 2]], dims=['a', 'b'])}, {'a': DataArray([0, 1], dims=['c']), 'b': DataArray([0, 1], dims=['c'])}]
        multiple_indexing(indexers2)
        indexers3 = [{'dim1': DataArray([[0, 7], [2, 6], [3, 5]], dims=['a', 'b']), 'dim3': slice(None, 10)}]
        multiple_indexing(indexers3)
        indexers4 = [{'dim3': 0}, {'dim1': DataArray([[0, 7], [2, 6], [3, 5]], dims=['a', 'b'])}, {'a': slice(None, None, 2)}]
        multiple_indexing(indexers4)
        indexers5 = [{'dim3': 0}, {'dim1': DataArray([[0, 7], [2, 6], [3, 5]], dims=['a', 'b'])}, {'a': 1, 'b': 0}]
        multiple_indexing(indexers5)

    def test_vectorized_indexing_negative_step(self) -> None:
        open_kwargs: dict[str, Any] | None
        if has_dask:
            open_kwargs = {'chunks': {}}
        else:
            open_kwargs = None
        in_memory = create_test_data()

        def multiple_indexing(indexers):
            with self.roundtrip(in_memory, open_kwargs=open_kwargs) as on_disk:
                actual = on_disk['var3']
                expected = in_memory['var3']
                for ind in indexers:
                    actual = actual.isel(ind)
                    expected = expected.isel(ind)
                    assert not actual.variable._in_memory
                assert_identical(expected, actual.load())
        indexers = [{'dim1': DataArray([[0, 7], [2, 6], [3, 5]], dims=['a', 'b']), 'dim3': slice(-1, 1, -1)}]
        multiple_indexing(indexers)
        indexers = [{'dim1': DataArray([[0, 7], [2, 6], [3, 5]], dims=['a', 'b']), 'dim3': slice(-1, 1, -2)}]
        multiple_indexing(indexers)

    def test_outer_indexing_reversed(self) -> None:
        ds = xr.Dataset({'z': (('t', 'p', 'y', 'x'), np.ones((1, 1, 31, 40)))})
        with self.roundtrip(ds) as on_disk:
            subset = on_disk.isel(t=[0], p=0).z[:, ::10, ::10][:, ::-1, :]
            assert subset.sizes == subset.load().sizes

    def test_isel_dataarray(self) -> None:
        in_memory = create_test_data()
        with self.roundtrip(in_memory) as on_disk:
            expected = in_memory.isel(dim2=in_memory['dim2'] < 3)
            actual = on_disk.isel(dim2=on_disk['dim2'] < 3)
            assert_identical(expected, actual)

    def validate_array_type(self, ds):

        def find_and_validate_array(obj):
            if hasattr(obj, 'array'):
                if isinstance(obj.array, indexing.ExplicitlyIndexed):
                    find_and_validate_array(obj.array)
                elif isinstance(obj.array, np.ndarray):
                    assert isinstance(obj, indexing.NumpyIndexingAdapter)
                elif isinstance(obj.array, dask_array_type):
                    assert isinstance(obj, indexing.DaskIndexingAdapter)
                elif isinstance(obj.array, pd.Index):
                    assert isinstance(obj, indexing.PandasIndexingAdapter)
                else:
                    raise TypeError(f'{type(obj.array)} is wrapped by {type(obj)}')
        for k, v in ds.variables.items():
            find_and_validate_array(v._data)

    def test_array_type_after_indexing(self) -> None:
        in_memory = create_test_data()
        with self.roundtrip(in_memory) as on_disk:
            self.validate_array_type(on_disk)
            indexers = {'dim1': [1, 2, 0], 'dim2': [3, 2, 0, 3], 'dim3': np.arange(5)}
            expected = in_memory.isel(indexers)
            actual = on_disk.isel(**indexers)
            assert_identical(expected, actual)
            self.validate_array_type(actual)
            actual = on_disk.isel(**indexers)
            assert_identical(expected, actual)
            self.validate_array_type(actual)

    def test_dropna(self) -> None:
        a = np.random.randn(4, 3)
        a[1, 1] = np.nan
        in_memory = xr.Dataset({'a': (('y', 'x'), a)}, coords={'y': np.arange(4), 'x': np.arange(3)})
        assert_identical(in_memory.dropna(dim='x'), in_memory.isel(x=slice(None, None, 2)))
        with self.roundtrip(in_memory) as on_disk:
            self.validate_array_type(on_disk)
            expected = in_memory.dropna(dim='x')
            actual = on_disk.dropna(dim='x')
            assert_identical(expected, actual)

    def test_ondisk_after_print(self) -> None:
        """Make sure print does not load file into memory"""
        in_memory = create_test_data()
        with self.roundtrip(in_memory) as on_disk:
            repr(on_disk)
            assert not on_disk['var1']._in_memory