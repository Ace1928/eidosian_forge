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
def test_reindex_with_nans(self):
    df = DataFrame([[1, 2], [3, 4], [np.nan, np.nan], [7, 8], [9, 10]], columns=['a', 'b'], index=[100.0, 101.0, np.nan, 102.0, 103.0])
    result = df.reindex(index=[101.0, 102.0, 103.0])
    expected = df.iloc[[1, 3, 4]]
    tm.assert_frame_equal(result, expected)
    result = df.reindex(index=[103.0])
    expected = df.iloc[[4]]
    tm.assert_frame_equal(result, expected)
    result = df.reindex(index=[101.0])
    expected = df.iloc[[1]]
    tm.assert_frame_equal(result, expected)