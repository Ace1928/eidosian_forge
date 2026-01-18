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
def test_reindex_multi_categorical_time(self):
    midx = MultiIndex.from_product([Categorical(['a', 'b', 'c']), Categorical(date_range('2012-01-01', periods=3, freq='h'))])
    df = DataFrame({'a': range(len(midx))}, index=midx)
    df2 = df.iloc[[0, 1, 2, 3, 4, 5, 6, 8]]
    result = df2.reindex(midx)
    expected = DataFrame({'a': [0, 1, 2, 3, 4, 5, 6, np.nan, 8]}, index=midx)
    tm.assert_frame_equal(result, expected)