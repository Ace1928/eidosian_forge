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
def test_pytables_native_read(datapath):
    with ensure_clean_store(datapath('io', 'data', 'legacy_hdf/pytables_native.h5'), mode='r') as store:
        d2 = store['detector/readout']
    assert isinstance(d2, DataFrame)