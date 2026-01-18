import contextlib
import datetime as dt
import hashlib
import tempfile
import time
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.pytables.common import (
from pandas.io.pytables import (
def test_to_hdf_with_min_itemsize(tmp_path, setup_path):
    path = tmp_path / setup_path
    df = DataFrame({'A': [0.0, 1.0, 2.0, 3.0, 4.0], 'B': [0.0, 1.0, 0.0, 1.0, 0.0], 'C': Index(['foo1', 'foo2', 'foo3', 'foo4', 'foo5'], dtype=object), 'D': date_range('20130101', periods=5)}).set_index('C')
    df.to_hdf(path, key='ss3', format='table', min_itemsize={'index': 6})
    df2 = df.copy().reset_index().assign(C='longer').set_index('C')
    df2.to_hdf(path, key='ss3', append=True, format='table')
    tm.assert_frame_equal(read_hdf(path, 'ss3'), concat([df, df2]))
    df['B'].to_hdf(path, key='ss4', format='table', min_itemsize={'index': 6})
    df2['B'].to_hdf(path, key='ss4', append=True, format='table')
    tm.assert_series_equal(read_hdf(path, 'ss4'), concat([df['B'], df2['B']]))