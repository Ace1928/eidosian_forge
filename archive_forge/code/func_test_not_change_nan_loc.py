from collections import namedtuple
from datetime import (
import re
from dateutil.tz import gettz
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import index as libindex
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.core.indexing import _one_ellipsis_message
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
@pytest.mark.parametrize('series, new_series, expected_ser', [[[np.nan, np.nan, 'b'], ['a', np.nan, np.nan], [False, True, True]], [[np.nan, 'b'], ['a', np.nan], [False, True]]])
def test_not_change_nan_loc(series, new_series, expected_ser):
    df = DataFrame({'A': series})
    df.loc[:, 'A'] = new_series
    expected = DataFrame({'A': expected_ser})
    tm.assert_frame_equal(df.isna(), expected)
    tm.assert_frame_equal(df.notna(), ~expected)