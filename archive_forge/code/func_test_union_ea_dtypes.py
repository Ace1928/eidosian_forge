from datetime import datetime
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.cast import find_common_type
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
def test_union_ea_dtypes(self, any_numeric_ea_and_arrow_dtype):
    idx = Index([1, 2, 3], dtype=any_numeric_ea_and_arrow_dtype)
    idx2 = Index([3, 4, 5], dtype=any_numeric_ea_and_arrow_dtype)
    result = idx.union(idx2)
    expected = Index([1, 2, 3, 4, 5], dtype=any_numeric_ea_and_arrow_dtype)
    tm.assert_index_equal(result, expected)