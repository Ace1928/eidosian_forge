from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
import pandas.tests.extension.base as eb
from packaging.version import Version
from datashader.datatypes import RaggedDtype, RaggedArray
from pandas.tests.extension.conftest import *  # noqa (fixture import)
class TestRaggedGetitem(eb.BaseGetitemTests):

    def test_get(self, data):
        s = pd.Series(data, index=[2 * i for i in range(len(data))])
        np.testing.assert_array_equal(s.get(4), s.iloc[2])
        result = s.get([4, 6])
        expected = s.iloc[[2, 3]]
        pd.testing.assert_series_equal(result, expected)
        result = s.get(slice(2))
        expected = s.iloc[[0, 1]]
        pd.testing.assert_series_equal(result, expected)
        assert s.get(-1) is None
        assert s.get(s.index.max() + 1) is None
        s = pd.Series(data[:6], index=list('abcdef'))
        np.testing.assert_array_equal(s.get('c'), s.iloc[2])
        result = s.get(slice('b', 'd'))
        expected = s.iloc[[1, 2, 3]]
        pd.testing.assert_series_equal(result, expected)
        result = s.get('Z')
        assert result is None

    def test_take_sequence(self, data):
        result = pd.Series(data)[[0, 1, 3]]
        np.testing.assert_array_equal(result.iloc[0], data[0])
        np.testing.assert_array_equal(result.iloc[1], data[1])
        np.testing.assert_array_equal(result.iloc[2], data[3])

    def test_take(self, data, na_value, na_cmp):
        result = data.take([0, -1])
        np.testing.assert_array_equal(result.dtype, data.dtype)
        np.testing.assert_array_equal(result[0], data[0])
        np.testing.assert_array_equal(result[1], data[-1])
        result = data.take([0, -1], allow_fill=True, fill_value=na_value)
        np.testing.assert_array_equal(result[0], data[0])
        assert na_cmp(result[1], na_value)
        with pytest.raises(IndexError, match='out of bounds'):
            data.take([len(data) + 1])

    def test_item(self, data):
        s = pd.Series(data)
        result = s[:1].item()
        np.testing.assert_array_equal(result, data[0])
        msg = 'can only convert an array of size 1 to a Python scalar'
        with pytest.raises(ValueError, match=msg):
            s[:0].item()
        with pytest.raises(ValueError, match=msg):
            s.item()

    @pytest.mark.skip(reason='Ellipsis not supported in RaggedArray.__getitem__')
    def test_getitem_ellipsis_and_slice(self, data):
        pass

    @pytest.mark.skip(reason='RaggedArray.__getitem__ raises a different error message')
    def test_getitem_invalid(self, data):
        pass

    @pytest.mark.skip(reason="Can't autoconvert ragged array to numpy array")
    def test_getitem_series_integer_with_missing_raises(self, data, idx):
        pass