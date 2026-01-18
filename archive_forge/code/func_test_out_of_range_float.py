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
def test_out_of_range_float(self):
    original = DataFrame({'ColumnOk': [0.0, np.finfo(np.float32).eps, np.finfo(np.float32).max / 10.0], 'ColumnTooBig': [0.0, np.finfo(np.float32).eps, np.finfo(np.float32).max]})
    original.index.name = 'index'
    for col in original:
        original[col] = original[col].astype(np.float32)
    with tm.ensure_clean() as path:
        original.to_stata(path)
        reread = read_stata(path)
    original['ColumnTooBig'] = original['ColumnTooBig'].astype(np.float64)
    expected = original.copy()
    expected.index = expected.index.astype(np.int32)
    tm.assert_frame_equal(reread.set_index('index'), expected)