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
def test_read_nokey_empty(tmp_path, setup_path):
    path = tmp_path / setup_path
    store = HDFStore(path)
    store.close()
    msg = re.escape('Dataset(s) incompatible with Pandas data types, not table, or no datasets found in HDF5 file.')
    with pytest.raises(ValueError, match=msg):
        read_hdf(path)