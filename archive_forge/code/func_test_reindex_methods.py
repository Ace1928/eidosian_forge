from datetime import (
import inspect
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
@pytest.mark.parametrize('method,expected_values', [('nearest', [0, 1, 1, 2]), ('pad', [np.nan, 0, 1, 1]), ('backfill', [0, 1, 2, 2])])
def test_reindex_methods(self, method, expected_values):
    df = DataFrame({'x': list(range(5))})
    target = np.array([-0.1, 0.9, 1.1, 1.5])
    expected = DataFrame({'x': expected_values}, index=target)
    actual = df.reindex(target, method=method)
    tm.assert_frame_equal(expected, actual)
    actual = df.reindex(target, method=method, tolerance=1)
    tm.assert_frame_equal(expected, actual)
    actual = df.reindex(target, method=method, tolerance=[1, 1, 1, 1])
    tm.assert_frame_equal(expected, actual)
    e2 = expected[::-1]
    actual = df.reindex(target[::-1], method=method)
    tm.assert_frame_equal(e2, actual)
    new_order = [3, 0, 2, 1]
    e2 = expected.iloc[new_order]
    actual = df.reindex(target[new_order], method=method)
    tm.assert_frame_equal(e2, actual)
    switched_method = 'pad' if method == 'backfill' else 'backfill' if method == 'pad' else method
    actual = df[::-1].reindex(target, method=switched_method)
    tm.assert_frame_equal(expected, actual)