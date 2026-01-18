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
def test_writer_117(self):
    original = DataFrame(data=[['string', 'object', 1, 1, 1, 1.1, 1.1, np.datetime64('2003-12-25'), 'a', 'a' * 2045, 'a' * 5000, 'a'], ['string-1', 'object-1', 1, 1, 1, 1.1, 1.1, np.datetime64('2003-12-26'), 'b', 'b' * 2045, '', '']], columns=['string', 'object', 'int8', 'int16', 'int32', 'float32', 'float64', 'datetime', 's1', 's2045', 'srtl', 'forced_strl'])
    original['object'] = Series(original['object'], dtype=object)
    original['int8'] = Series(original['int8'], dtype=np.int8)
    original['int16'] = Series(original['int16'], dtype=np.int16)
    original['int32'] = original['int32'].astype(np.int32)
    original['float32'] = Series(original['float32'], dtype=np.float32)
    original.index.name = 'index'
    original.index = original.index.astype(np.int32)
    copy = original.copy()
    with tm.ensure_clean() as path:
        original.to_stata(path, convert_dates={'datetime': 'tc'}, convert_strl=['forced_strl'], version=117)
        written_and_read_again = self.read_dta(path)
        tm.assert_frame_equal(written_and_read_again.set_index('index'), original, check_index_type=False)
        tm.assert_frame_equal(original, copy)