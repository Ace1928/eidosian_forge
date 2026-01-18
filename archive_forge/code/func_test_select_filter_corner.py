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
def test_select_filter_corner(setup_path):
    df = DataFrame(np.random.default_rng(2).standard_normal((50, 100)))
    df.index = [f'{c:3d}' for c in df.index]
    df.columns = [f'{c:3d}' for c in df.columns]
    with ensure_clean_store(setup_path) as store:
        store.put('frame', df, format='table')
        crit = 'columns=df.columns[:75]'
        result = store.select('frame', [crit])
        tm.assert_frame_equal(result, df.loc[:, df.columns[:75]])
        crit = 'columns=df.columns[:75:2]'
        result = store.select('frame', [crit])
        tm.assert_frame_equal(result, df.loc[:, df.columns[:75:2]])