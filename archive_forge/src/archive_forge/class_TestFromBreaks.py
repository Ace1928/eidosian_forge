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
class TestFromBreaks(ConstructorTests):
    """Tests specific to IntervalIndex.from_breaks"""

    @pytest.fixture
    def constructor(self):
        return IntervalIndex.from_breaks

    def get_kwargs_from_breaks(self, breaks, closed='right'):
        """
        converts intervals in breaks format to a dictionary of kwargs to
        specific to the format expected by IntervalIndex.from_breaks
        """
        return {'breaks': breaks}

    def test_constructor_errors(self):
        data = Categorical(list('01234abcde'), ordered=True)
        msg = 'category, object, and string subtypes are not supported for IntervalIndex'
        with pytest.raises(TypeError, match=msg):
            IntervalIndex.from_breaks(data)

    def test_length_one(self):
        """breaks of length one produce an empty IntervalIndex"""
        breaks = [0]
        result = IntervalIndex.from_breaks(breaks)
        expected = IntervalIndex.from_breaks([])
        tm.assert_index_equal(result, expected)

    def test_left_right_dont_share_data(self):
        breaks = np.arange(5)
        result = IntervalIndex.from_breaks(breaks)._data
        assert result._left.base is None or result._left.base is not result._right.base