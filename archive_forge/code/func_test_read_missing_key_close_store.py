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
def test_read_missing_key_close_store(tmp_path, setup_path):
    path = tmp_path / setup_path
    df = DataFrame({'a': range(2), 'b': range(2)})
    df.to_hdf(path, key='k1')
    with pytest.raises(KeyError, match="'No object named k2 in the file'"):
        read_hdf(path, 'k2')
    df.to_hdf(path, key='k2')