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
def test_reindex_multi(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)))
    result = df.reindex(index=range(4), columns=range(4))
    expected = df.reindex(list(range(4))).reindex(columns=range(4))
    tm.assert_frame_equal(result, expected)
    df = DataFrame(np.random.default_rng(2).integers(0, 10, (3, 3)))
    result = df.reindex(index=range(4), columns=range(4))
    expected = df.reindex(list(range(4))).reindex(columns=range(4))
    tm.assert_frame_equal(result, expected)
    df = DataFrame(np.random.default_rng(2).integers(0, 10, (3, 3)))
    result = df.reindex(index=range(2), columns=range(2))
    expected = df.reindex(range(2)).reindex(columns=range(2))
    tm.assert_frame_equal(result, expected)
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)) + 1j, columns=['a', 'b', 'c'])
    result = df.reindex(index=[0, 1], columns=['a', 'b'])
    expected = df.reindex([0, 1]).reindex(columns=['a', 'b'])
    tm.assert_frame_equal(result, expected)