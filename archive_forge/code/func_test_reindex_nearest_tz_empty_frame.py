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
def test_reindex_nearest_tz_empty_frame(self):
    dti = pd.DatetimeIndex(['2016-06-26 14:27:26+00:00'])
    df = DataFrame(index=pd.DatetimeIndex(['2016-07-04 14:00:59+00:00']))
    expected = DataFrame(index=dti)
    result = df.reindex(dti, method='nearest')
    tm.assert_frame_equal(result, expected)