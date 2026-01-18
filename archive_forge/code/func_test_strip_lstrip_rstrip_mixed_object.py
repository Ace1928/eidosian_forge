from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.strings.accessor import StringMethods
from pandas.tests.strings import object_pyarrow_numpy
@pytest.mark.parametrize('method, exp', [['strip', ['aa', np.nan, 'bb']], ['lstrip', ['aa  ', np.nan, 'bb \t\n']], ['rstrip', ['  aa', np.nan, ' bb']]])
def test_strip_lstrip_rstrip_mixed_object(method, exp):
    ser = Series(['  aa  ', np.nan, ' bb \t\n', True, datetime.today(), None, 1, 2.0])
    result = getattr(ser.str, method)()
    expected = Series(exp + [np.nan, np.nan, None, np.nan, np.nan], dtype=object)
    tm.assert_series_equal(result, expected)