import datetime
import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.util import _test_decorators as td
def test_round_trip_equals(tmp_path, setup_path):
    df = DataFrame({'B': [1, 2], 'A': ['x', 'y']})
    path = tmp_path / setup_path
    df.to_hdf(path, key='df', format='table')
    other = read_hdf(path, 'df')
    tm.assert_frame_equal(df, other)
    assert df.equals(other)
    assert other.equals(df)