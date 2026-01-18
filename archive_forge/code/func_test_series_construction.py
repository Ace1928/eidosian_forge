from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
import pandas.tests.extension.base as eb
from packaging.version import Version
from datashader.datatypes import RaggedDtype, RaggedArray
from pandas.tests.extension.conftest import *  # noqa (fixture import)
def test_series_construction():
    arg = [[0, 1], [1.0, 2, 3.0, 4], None, [-1, -2]] * 2
    rs = pd.Series(arg, dtype='Ragged[int64]')
    ra = rs.array
    expected = RaggedArray(arg, dtype='int64')
    assert_ragged_arrays_equal(ra, expected)