from __future__ import annotations
import re
import warnings
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.filterwarnings('ignore:Period with BDay freq is deprecated:FutureWarning')
def test_take_fill_str(self, arr1d):
    result = arr1d.take([-1, 1], allow_fill=True, fill_value=str(arr1d[-1]))
    expected = arr1d[[-1, 1]]
    tm.assert_equal(result, expected)
    msg = f"value should be a '{arr1d._scalar_type.__name__}' or 'NaT'. Got"
    with pytest.raises(TypeError, match=msg):
        arr1d.take([-1, 1], allow_fill=True, fill_value='foo')