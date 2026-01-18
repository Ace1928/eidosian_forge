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
@pytest.mark.parametrize('dtype_backend', ['numpy_nullable', pytest.param('pyarrow', marks=td.skip_if_no('pyarrow'))])
def test_read_write_ea_dtypes(self, dtype_backend):
    df = DataFrame({'a': [1, 2, None], 'b': ['a', 'b', 'c'], 'c': [True, False, None], 'd': [1.5, 2.5, 3.5], 'e': pd.date_range('2020-12-31', periods=3, freq='D')}, index=pd.Index([0, 1, 2], name='index'))
    df = df.convert_dtypes(dtype_backend=dtype_backend)
    df.to_stata('test_stata.dta', version=118)
    with tm.ensure_clean() as path:
        df.to_stata(path)
        written_and_read_again = self.read_dta(path)
    expected = DataFrame({'a': [1, 2, np.nan], 'b': ['a', 'b', 'c'], 'c': [1.0, 0, np.nan], 'd': [1.5, 2.5, 3.5], 'e': pd.date_range('2020-12-31', periods=3, freq='D')}, index=pd.Index([0, 1, 2], name='index', dtype=np.int32))
    tm.assert_frame_equal(written_and_read_again.set_index('index'), expected)