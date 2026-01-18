import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_update_datetime_tz_in_place(self, using_copy_on_write, warn_copy_on_write):
    result = DataFrame([pd.Timestamp('2019', tz='UTC')])
    orig = result.copy()
    view = result[:]
    with tm.assert_produces_warning(FutureWarning if warn_copy_on_write else None, match='Setting a value'):
        result.update(result + pd.Timedelta(days=1))
    expected = DataFrame([pd.Timestamp('2019-01-02', tz='UTC')])
    tm.assert_frame_equal(result, expected)
    if not using_copy_on_write:
        tm.assert_frame_equal(view, expected)
    else:
        tm.assert_frame_equal(view, orig)