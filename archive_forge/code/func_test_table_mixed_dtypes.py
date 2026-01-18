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
def test_table_mixed_dtypes(setup_path):
    df = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=Index(list('ABCD'), dtype=object), index=Index([f'i-{i}' for i in range(30)], dtype=object))
    df['obj1'] = 'foo'
    df['obj2'] = 'bar'
    df['bool1'] = df['A'] > 0
    df['bool2'] = df['B'] > 0
    df['bool3'] = True
    df['int1'] = 1
    df['int2'] = 2
    df['timestamp1'] = Timestamp('20010102').as_unit('ns')
    df['timestamp2'] = Timestamp('20010103').as_unit('ns')
    df['datetime1'] = Timestamp('20010102').as_unit('ns')
    df['datetime2'] = Timestamp('20010103').as_unit('ns')
    df.loc[df.index[3:6], ['obj1']] = np.nan
    df = df._consolidate()
    with ensure_clean_store(setup_path) as store:
        store.append('df1_mixed', df)
        tm.assert_frame_equal(store.select('df1_mixed'), df)