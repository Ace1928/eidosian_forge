from datetime import (
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
def test_union_freq_both_none(self, sort):
    expected = bdate_range('20150101', periods=10)
    expected._data.freq = None
    result = expected.union(expected, sort=sort)
    tm.assert_index_equal(result, expected)
    assert result.freq is None