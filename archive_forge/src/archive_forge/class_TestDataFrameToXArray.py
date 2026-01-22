import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
class TestDataFrameToXArray:

    @pytest.fixture
    def df(self):
        return DataFrame({'a': list('abcd'), 'b': list(range(1, 5)), 'c': np.arange(3, 7).astype('u1'), 'd': np.arange(4.0, 8.0, dtype='float64'), 'e': [True, False, True, False], 'f': Categorical(list('abcd')), 'g': date_range('20130101', periods=4), 'h': date_range('20130101', periods=4, tz='US/Eastern')})

    def test_to_xarray_index_types(self, index_flat, df, using_infer_string):
        index = index_flat
        if len(index) == 0:
            pytest.skip("Test doesn't make sense for empty index")
        from xarray import Dataset
        df.index = index[:4]
        df.index.name = 'foo'
        df.columns.name = 'bar'
        result = df.to_xarray()
        assert result.sizes['foo'] == 4
        assert len(result.coords) == 1
        assert len(result.data_vars) == 8
        tm.assert_almost_equal(list(result.coords.keys()), ['foo'])
        assert isinstance(result, Dataset)
        expected = df.copy()
        expected['f'] = expected['f'].astype(object if not using_infer_string else 'string[pyarrow_numpy]')
        expected.columns.name = None
        tm.assert_frame_equal(result.to_dataframe(), expected)

    def test_to_xarray_empty(self, df):
        from xarray import Dataset
        df.index.name = 'foo'
        result = df[0:0].to_xarray()
        assert result.sizes['foo'] == 0
        assert isinstance(result, Dataset)

    def test_to_xarray_with_multiindex(self, df, using_infer_string):
        from xarray import Dataset
        df.index = MultiIndex.from_product([['a'], range(4)], names=['one', 'two'])
        result = df.to_xarray()
        assert result.sizes['one'] == 1
        assert result.sizes['two'] == 4
        assert len(result.coords) == 2
        assert len(result.data_vars) == 8
        tm.assert_almost_equal(list(result.coords.keys()), ['one', 'two'])
        assert isinstance(result, Dataset)
        result = result.to_dataframe()
        expected = df.copy()
        expected['f'] = expected['f'].astype(object if not using_infer_string else 'string[pyarrow_numpy]')
        expected.columns.name = None
        tm.assert_frame_equal(result, expected)