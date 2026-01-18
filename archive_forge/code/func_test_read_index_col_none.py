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
@pytest.mark.parametrize('version', [114, 117, 118, 119, None])
def test_read_index_col_none(self, version):
    df = DataFrame({'a': range(5), 'b': ['b1', 'b2', 'b3', 'b4', 'b5']})
    with tm.ensure_clean() as path:
        df.to_stata(path, write_index=False, version=version)
        read_df = read_stata(path)
    assert isinstance(read_df.index, pd.RangeIndex)
    expected = df.copy()
    expected['a'] = expected['a'].astype(np.int32)
    tm.assert_frame_equal(read_df, expected, check_index_type=True)