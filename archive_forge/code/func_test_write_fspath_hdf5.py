import codecs
import errno
from functools import partial
from io import (
import mmap
import os
from pathlib import Path
import pickle
import tempfile
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
import pandas.io.common as icom
def test_write_fspath_hdf5(self):
    pytest.importorskip('tables')
    df = pd.DataFrame({'A': [1, 2]})
    p1 = tm.ensure_clean('string')
    p2 = tm.ensure_clean('fspath')
    with p1 as string, p2 as fspath:
        mypath = CustomFSPath(fspath)
        df.to_hdf(mypath, key='bar')
        df.to_hdf(string, key='bar')
        result = pd.read_hdf(fspath, key='bar')
        expected = pd.read_hdf(string, key='bar')
    tm.assert_frame_equal(result, expected)