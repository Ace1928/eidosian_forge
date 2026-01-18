from datetime import datetime
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.cast import find_common_type
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
def test_union_string_array(self, any_string_dtype):
    idx1 = Index(['a'], dtype=any_string_dtype)
    idx2 = Index(['b'], dtype=any_string_dtype)
    result = idx1.union(idx2)
    expected = Index(['a', 'b'], dtype=any_string_dtype)
    tm.assert_index_equal(result, expected)