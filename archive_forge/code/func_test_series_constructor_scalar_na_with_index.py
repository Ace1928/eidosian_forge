import collections
import operator
import sys
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.json.array import (
@pytest.mark.xfail(reason='RecursionError, GH-33900')
def test_series_constructor_scalar_na_with_index(self, dtype, na_value):
    rec_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(100)
        super().test_series_constructor_scalar_na_with_index(dtype, na_value)
    finally:
        sys.setrecursionlimit(rec_limit)