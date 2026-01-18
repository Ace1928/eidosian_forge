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
def test_reindex_limit(self):
    data = [['A', 'A', 'A'], ['B', 'B', 'B'], ['C', 'C', 'C'], ['D', 'D', 'D']]
    exp_data = [['A', 'A', 'A'], ['B', 'B', 'B'], ['C', 'C', 'C'], ['D', 'D', 'D'], ['D', 'D', 'D'], [np.nan, np.nan, np.nan]]
    df = DataFrame(data)
    result = df.reindex([0, 1, 2, 3, 4, 5], method='ffill', limit=1)
    expected = DataFrame(exp_data)
    tm.assert_frame_equal(result, expected)