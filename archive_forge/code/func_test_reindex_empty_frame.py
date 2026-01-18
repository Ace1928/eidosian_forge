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
@pytest.mark.parametrize('kwargs', [{'method': 'pad', 'tolerance': timedelta(seconds=9)}, {'method': 'backfill', 'tolerance': timedelta(seconds=9)}, {'method': 'nearest'}, {'method': None}])
def test_reindex_empty_frame(self, kwargs):
    idx = date_range(start='2020', freq='30s', periods=3)
    df = DataFrame([], index=Index([], name='time'), columns=['a'])
    result = df.reindex(idx, **kwargs)
    expected = DataFrame({'a': [np.nan] * 3}, index=idx, dtype=object)
    tm.assert_frame_equal(result, expected)