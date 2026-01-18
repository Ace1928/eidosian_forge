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
def test_to_csv_from_csv3(self):
    with tm.ensure_clean('__tmp_to_csv_from_csv3__') as path:
        df1 = DataFrame(np.random.default_rng(2).standard_normal((3, 1)))
        df2 = DataFrame(np.random.default_rng(2).standard_normal((3, 1)))
        df1.to_csv(path)
        df2.to_csv(path, mode='a', header=False)
        xp = pd.concat([df1, df2])
        rs = read_csv(path, index_col=0)
        rs.columns = [int(label) for label in rs.columns]
        xp.columns = [int(label) for label in xp.columns]
        tm.assert_frame_equal(xp, rs)