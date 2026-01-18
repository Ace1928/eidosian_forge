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
def test_store_dropna(tmp_path, setup_path):
    df_with_missing = DataFrame({'col1': [0.0, np.nan, 2.0], 'col2': [1.0, np.nan, np.nan]}, index=list('abc'))
    df_without_missing = DataFrame({'col1': [0.0, 2.0], 'col2': [1.0, np.nan]}, index=list('ac'))
    path = tmp_path / setup_path
    df_with_missing.to_hdf(path, key='df', format='table')
    reloaded = read_hdf(path, 'df')
    tm.assert_frame_equal(df_with_missing, reloaded)
    path = tmp_path / setup_path
    df_with_missing.to_hdf(path, key='df', format='table', dropna=False)
    reloaded = read_hdf(path, 'df')
    tm.assert_frame_equal(df_with_missing, reloaded)
    path = tmp_path / setup_path
    df_with_missing.to_hdf(path, key='df', format='table', dropna=True)
    reloaded = read_hdf(path, 'df')
    tm.assert_frame_equal(df_without_missing, reloaded)