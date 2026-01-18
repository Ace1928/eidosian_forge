import bz2
import datetime as dt
from datetime import datetime
import gzip
import io
import os
import struct
import tarfile
import zipfile
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import CategoricalDtype
import pandas._testing as tm
from pandas.core.frame import (
from pandas.io.parsers import read_csv
from pandas.io.stata import (
@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.parametrize('file', ['stata1_117', 'stata2_117', 'stata3_117', 'stata4_117', 'stata5_117', 'stata6_117', 'stata7_117', 'stata8_117', 'stata9_117', 'stata10_117', 'stata11_117'])
@pytest.mark.parametrize('chunksize', [1, 2])
@pytest.mark.parametrize('convert_categoricals', [False, True])
@pytest.mark.parametrize('convert_dates', [False, True])
def test_read_chunks_117(self, file, chunksize, convert_categoricals, convert_dates, datapath):
    fname = datapath('io', 'data', 'stata', f'{file}.dta')
    parsed = read_stata(fname, convert_categoricals=convert_categoricals, convert_dates=convert_dates)
    with read_stata(fname, iterator=True, convert_categoricals=convert_categoricals, convert_dates=convert_dates) as itr:
        pos = 0
        for j in range(5):
            try:
                chunk = itr.read(chunksize)
            except StopIteration:
                break
            from_frame = parsed.iloc[pos:pos + chunksize, :].copy()
            from_frame = self._convert_categorical(from_frame)
            tm.assert_frame_equal(from_frame, chunk, check_dtype=False, check_datetimelike_compat=True)
            pos += chunksize