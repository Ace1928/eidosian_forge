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
def test_scalar_from_string(self, arr1d):
    result = arr1d._scalar_from_string(str(arr1d[0]))
    assert result == arr1d[0]