from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
def test_series_indexing_zerodim_np_array(self):
    s = Series([1, 2])
    result = s.iloc[np.array(0)]
    assert result == 1