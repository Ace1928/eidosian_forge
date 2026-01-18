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
def test_read_index_error_close_store(tmp_path, setup_path):
    path = tmp_path / setup_path
    df = DataFrame({'A': [], 'B': []}, index=[])
    df.to_hdf(path, key='k1')
    with pytest.raises(IndexError, match='list index out of range'):
        read_hdf(path, 'k1', stop=0)
    df.to_hdf(path, key='k1')