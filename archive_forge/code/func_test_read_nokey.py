from contextlib import closing
from pathlib import Path
import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.util import _test_decorators as td
from pandas.io.pytables import TableIterator
def test_read_nokey(tmp_path, setup_path):
    df = DataFrame(np.random.default_rng(2).random((4, 5)), index=list('abcd'), columns=list('ABCDE'))
    path = tmp_path / setup_path
    df.to_hdf(path, key='df', mode='a')
    reread = read_hdf(path)
    tm.assert_frame_equal(df, reread)
    df.to_hdf(path, key='df2', mode='a')
    msg = 'key must be provided when HDF5 file contains multiple datasets.'
    with pytest.raises(ValueError, match=msg):
        read_hdf(path)