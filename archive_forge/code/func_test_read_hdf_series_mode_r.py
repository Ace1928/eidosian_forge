from contextlib import closing
from pathlib import Path
import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.util import _test_decorators as td
from pandas.io.pytables import TableIterator
@pytest.mark.parametrize('format', ['fixed', 'table'])
def test_read_hdf_series_mode_r(tmp_path, format, setup_path):
    series = Series(range(10), dtype=np.float64)
    path = tmp_path / setup_path
    series.to_hdf(path, key='data', format=format)
    result = read_hdf(path, key='data', mode='r')
    tm.assert_series_equal(result, series)