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
@pytest.mark.slow
def test_to_csv_dtnat(self):

    def make_dtnat_arr(n, nnat=None):
        if nnat is None:
            nnat = int(n * 0.1)
        s = list(date_range('2000', freq='5min', periods=n))
        if nnat:
            for i in np.random.default_rng(2).integers(0, len(s), nnat):
                s[i] = NaT
            i = np.random.default_rng(2).integers(100)
            s[-i] = NaT
            s[i] = NaT
        return s
    chunksize = 1000
    s1 = make_dtnat_arr(chunksize + 5)
    s2 = make_dtnat_arr(chunksize + 5, 0)
    with tm.ensure_clean('1.csv') as pth:
        df = DataFrame({'a': s1, 'b': s2})
        df.to_csv(pth, chunksize=chunksize)
        recons = self.read_csv(pth).apply(to_datetime)
        tm.assert_frame_equal(df, recons, check_names=False)