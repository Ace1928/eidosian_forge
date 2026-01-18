import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('actual', [DataFrame(data=data, index=['a', 'a']), DataFrame(data=data, index=['a', 'b']), DataFrame(data=data, index=['a', 'b']).set_index([0, 1]), DataFrame(data=data, index=['a', 'a']).set_index([0, 1])])
def test_raise_on_drop_duplicate_index(self, actual):
    level = 0 if isinstance(actual.index, MultiIndex) else None
    msg = re.escape('"[\'c\'] not found in axis"')
    with pytest.raises(KeyError, match=msg):
        actual.drop('c', level=level, axis=0)
    with pytest.raises(KeyError, match=msg):
        actual.T.drop('c', level=level, axis=1)
    expected_no_err = actual.drop('c', axis=0, level=level, errors='ignore')
    tm.assert_frame_equal(expected_no_err, actual)
    expected_no_err = actual.T.drop('c', axis=1, level=level, errors='ignore')
    tm.assert_frame_equal(expected_no_err.T, actual)