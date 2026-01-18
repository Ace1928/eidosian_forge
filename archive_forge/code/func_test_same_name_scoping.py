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
def test_same_name_scoping(setup_path):
    with ensure_clean_store(setup_path) as store:
        df = DataFrame(np.random.default_rng(2).standard_normal((20, 2)), index=date_range('20130101', periods=20))
        store.put('df', df, format='table')
        expected = df[df.index > Timestamp('20130105')]
        result = store.select('df', 'index>datetime.datetime(2013,1,5)')
        tm.assert_frame_equal(result, expected)
        result = store.select('df', 'index>datetime.datetime(2013,1,5)')
        tm.assert_frame_equal(result, expected)
        result = store.select('df', 'index>datetime(2013,1,5)')
        tm.assert_frame_equal(result, expected)