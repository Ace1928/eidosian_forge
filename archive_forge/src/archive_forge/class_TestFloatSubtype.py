import re
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import (
from pandas import (
import pandas._testing as tm
class TestFloatSubtype(AstypeTests):
    """Tests specific to IntervalIndex with float subtype"""
    indexes = [interval_range(-10.0, 10.0, closed='neither'), IntervalIndex.from_arrays([-1.5, np.nan, 0.0, 0.0, 1.5], [-0.5, np.nan, 1.0, 1.0, 3.0], closed='both')]

    @pytest.fixture(params=indexes)
    def index(self, request):
        return request.param

    @pytest.mark.parametrize('subtype', ['int64', 'uint64'])
    def test_subtype_integer(self, subtype):
        index = interval_range(0.0, 10.0)
        dtype = IntervalDtype(subtype, 'right')
        result = index.astype(dtype)
        expected = IntervalIndex.from_arrays(index.left.astype(subtype), index.right.astype(subtype), closed=index.closed)
        tm.assert_index_equal(result, expected)
        msg = 'Cannot convert non-finite values \\(NA or inf\\) to integer'
        with pytest.raises(ValueError, match=msg):
            index.insert(0, np.nan).astype(dtype)

    @pytest.mark.parametrize('subtype', ['int64', 'uint64'])
    def test_subtype_integer_with_non_integer_borders(self, subtype):
        index = interval_range(0.0, 3.0, freq=0.25)
        dtype = IntervalDtype(subtype, 'right')
        result = index.astype(dtype)
        expected = IntervalIndex.from_arrays(index.left.astype(subtype), index.right.astype(subtype), closed=index.closed)
        tm.assert_index_equal(result, expected)

    def test_subtype_integer_errors(self):
        index = interval_range(-10.0, 10.0)
        dtype = IntervalDtype('uint64', 'right')
        msg = re.escape('Cannot convert interval[float64, right] to interval[uint64, right]; subtypes are incompatible')
        with pytest.raises(TypeError, match=msg):
            index.astype(dtype)

    @pytest.mark.parametrize('subtype', ['datetime64[ns]', 'timedelta64[ns]'])
    def test_subtype_datetimelike(self, index, subtype):
        dtype = IntervalDtype(subtype, 'right')
        msg = 'Cannot convert .* to .*; subtypes are incompatible'
        with pytest.raises(TypeError, match=msg):
            index.astype(dtype)