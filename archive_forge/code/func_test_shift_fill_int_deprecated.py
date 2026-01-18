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
def test_shift_fill_int_deprecated(self, arr1d):
    with pytest.raises(TypeError, match='value should be a'):
        arr1d.shift(1, fill_value=1)