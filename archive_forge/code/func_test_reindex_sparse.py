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
def test_reindex_sparse(self):
    df = DataFrame({'A': [0, 1], 'B': pd.array([0, 1], dtype=pd.SparseDtype('int64', 0))})
    result = df.reindex([0, 2])
    expected = DataFrame({'A': [0.0, np.nan], 'B': pd.array([0.0, np.nan], dtype=pd.SparseDtype('float64', 0.0))}, index=[0, 2])
    tm.assert_frame_equal(result, expected)