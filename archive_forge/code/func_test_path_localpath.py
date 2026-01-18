import contextlib
from datetime import datetime
import io
import os
from pathlib import Path
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import EmptyDataError
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.io.sas.sas7bdat import SAS7BDATReader
@td.skip_if_no('py.path')
@pytest.mark.slow
def test_path_localpath(self, dirpath, data_test_ix):
    from py.path import local as LocalPath
    expected, test_ix = data_test_ix
    for k in test_ix:
        fname = LocalPath(os.path.join(dirpath, f'test{k}.sas7bdat'))
        df = pd.read_sas(fname, encoding='utf-8')
        tm.assert_frame_equal(df, expected)