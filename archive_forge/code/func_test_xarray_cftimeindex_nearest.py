import array
import subprocess
import sys
import numpy as np
import pytest
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_xarray_cftimeindex_nearest():
    cftime = pytest.importorskip('cftime')
    xarray = pytest.importorskip('xarray')
    times = xarray.cftime_range('0001', periods=2)
    key = cftime.DatetimeGregorian(2000, 1, 1)
    result = times.get_indexer([key], method='nearest')
    expected = 1
    assert result == expected