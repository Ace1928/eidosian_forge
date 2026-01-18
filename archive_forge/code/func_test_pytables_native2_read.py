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
@pytest.mark.skipif(is_platform_windows(), reason='native2 read fails oddly on windows')
def test_pytables_native2_read(datapath):
    with ensure_clean_store(datapath('io', 'data', 'legacy_hdf', 'pytables_native2.h5'), mode='r') as store:
        str(store)
        d1 = store['detector']
    assert isinstance(d1, DataFrame)