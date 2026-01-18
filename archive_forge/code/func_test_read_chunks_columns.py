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
def test_read_chunks_columns(self, datapath):
    fname = datapath('io', 'data', 'stata', 'stata3_117.dta')
    columns = ['quarter', 'cpi', 'm1']
    chunksize = 2
    parsed = read_stata(fname, columns=columns)
    with read_stata(fname, iterator=True) as itr:
        pos = 0
        for j in range(5):
            chunk = itr.read(chunksize, columns=columns)
            if chunk is None:
                break
            from_frame = parsed.iloc[pos:pos + chunksize, :]
            tm.assert_frame_equal(from_frame, chunk, check_dtype=False)
            pos += chunksize