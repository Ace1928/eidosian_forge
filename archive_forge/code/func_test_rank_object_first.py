from datetime import (
import numpy as np
import pytest
from pandas._libs.algos import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('na_option,ascending,expected', [('bottom', True, [1.0, 2.0, 4.0, 3.0]), ('bottom', False, [1.0, 2.0, 4.0, 3.0]), ('top', True, [2.0, 3.0, 1.0, 4.0]), ('top', False, [2.0, 3.0, 1.0, 4.0])])
def test_rank_object_first(self, frame_or_series, na_option, ascending, expected, using_infer_string):
    obj = frame_or_series(['foo', 'foo', None, 'foo'])
    result = obj.rank(method='first', na_option=na_option, ascending=ascending)
    expected = frame_or_series(expected)
    if using_infer_string and isinstance(obj, Series):
        expected = expected.astype('uint64')
    tm.assert_equal(result, expected)