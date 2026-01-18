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
@pytest.mark.parametrize('as_index', [True, False])
def test_setitem_categorical(self, arr1d, as_index):
    expected = arr1d.copy()[::-1]
    if not isinstance(expected, PeriodArray):
        expected = expected._with_freq(None)
    cat = pd.Categorical(arr1d)
    if as_index:
        cat = pd.CategoricalIndex(cat)
    arr1d[:] = cat[::-1]
    tm.assert_equal(arr1d, expected)