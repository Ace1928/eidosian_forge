from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_unsigned_integer_dtype
from pandas.core.dtypes.dtypes import IntervalDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
import pandas.core.common as com
class TestFromArrays(ConstructorTests):
    """Tests specific to IntervalIndex.from_arrays"""

    @pytest.fixture
    def constructor(self):
        return IntervalIndex.from_arrays

    def get_kwargs_from_breaks(self, breaks, closed='right'):
        """
        converts intervals in breaks format to a dictionary of kwargs to
        specific to the format expected by IntervalIndex.from_arrays
        """
        return {'left': breaks[:-1], 'right': breaks[1:]}

    def test_constructor_errors(self):
        data = Categorical(list('01234abcde'), ordered=True)
        msg = 'category, object, and string subtypes are not supported for IntervalIndex'
        with pytest.raises(TypeError, match=msg):
            IntervalIndex.from_arrays(data[:-1], data[1:])
        left = [0, 1, 2]
        right = [2, 3]
        msg = 'left and right must have the same length'
        with pytest.raises(ValueError, match=msg):
            IntervalIndex.from_arrays(left, right)

    @pytest.mark.parametrize('left_subtype, right_subtype', [(np.int64, np.float64), (np.float64, np.int64)])
    def test_mixed_float_int(self, left_subtype, right_subtype):
        """mixed int/float left/right results in float for both sides"""
        left = np.arange(9, dtype=left_subtype)
        right = np.arange(1, 10, dtype=right_subtype)
        result = IntervalIndex.from_arrays(left, right)
        expected_left = Index(left, dtype=np.float64)
        expected_right = Index(right, dtype=np.float64)
        expected_subtype = np.float64
        tm.assert_index_equal(result.left, expected_left)
        tm.assert_index_equal(result.right, expected_right)
        assert result.dtype.subtype == expected_subtype

    @pytest.mark.parametrize('interval_cls', [IntervalArray, IntervalIndex])
    def test_from_arrays_mismatched_datetimelike_resos(self, interval_cls):
        left = date_range('2016-01-01', periods=3, unit='s')
        right = date_range('2017-01-01', periods=3, unit='ms')
        result = interval_cls.from_arrays(left, right)
        expected = interval_cls.from_arrays(left.as_unit('ms'), right)
        tm.assert_equal(result, expected)
        left2 = left - left[0]
        right2 = right - left[0]
        result2 = interval_cls.from_arrays(left2, right2)
        expected2 = interval_cls.from_arrays(left2.as_unit('ms'), right2)
        tm.assert_equal(result2, expected2)
        left3 = left.tz_localize('UTC')
        right3 = right.tz_localize('UTC')
        result3 = interval_cls.from_arrays(left3, right3)
        expected3 = interval_cls.from_arrays(left3.as_unit('ms'), right3)
        tm.assert_equal(result3, expected3)