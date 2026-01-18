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
def test_reindex_multiindex_ffill_added_rows(self):
    mi = MultiIndex.from_tuples([('a', 'b'), ('d', 'e')])
    df = DataFrame([[0, 7], [3, 4]], index=mi, columns=['x', 'y'])
    mi2 = MultiIndex.from_tuples([('a', 'b'), ('d', 'e'), ('h', 'i')])
    result = df.reindex(mi2, axis=0, method='ffill')
    expected = DataFrame([[0, 7], [3, 4], [3, 4]], index=mi2, columns=['x', 'y'])
    tm.assert_frame_equal(result, expected)