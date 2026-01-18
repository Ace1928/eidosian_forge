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
def test_read_from_pathlib_path(tmp_path, setup_path):
    expected = DataFrame(np.random.default_rng(2).random((4, 5)), index=list('abcd'), columns=list('ABCDE'))
    filename = tmp_path / setup_path
    path_obj = Path(filename)
    expected.to_hdf(path_obj, key='df', mode='a')
    actual = read_hdf(path_obj, key='df')
    tm.assert_frame_equal(expected, actual)