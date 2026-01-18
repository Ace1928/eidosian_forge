import csv
from io import StringIO
import os
import numpy as np
import pytest
from pandas.errors import ParserError
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
from pandas.io.common import get_handle
@pytest.mark.parametrize('cols', [None, ['b', 'a']])
def test_to_csv_new_dupe_cols(self, cols):
    chunksize = 5
    N = int(chunksize * 2.5)
    df = DataFrame(np.ones((N, 3)), index=Index([f'i-{i}' for i in range(N)], name='a'), columns=['a', 'a', 'b'])
    with tm.ensure_clean() as path:
        df.to_csv(path, columns=cols, chunksize=chunksize)
        rs_c = read_csv(path, index_col=0)
        if cols is not None:
            if df.columns.is_unique:
                rs_c.columns = cols
            else:
                indexer, missing = df.columns.get_indexer_non_unique(cols)
                rs_c.columns = df.columns.take(indexer)
            for c in cols:
                obj_df = df[c]
                obj_rs = rs_c[c]
                if isinstance(obj_df, Series):
                    tm.assert_series_equal(obj_df, obj_rs)
                else:
                    tm.assert_frame_equal(obj_df, obj_rs, check_names=False)
        else:
            rs_c.columns = df.columns
            tm.assert_frame_equal(df, rs_c, check_names=False)