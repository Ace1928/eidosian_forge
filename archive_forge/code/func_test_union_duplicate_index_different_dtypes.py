from datetime import datetime
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.cast import find_common_type
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
def test_union_duplicate_index_different_dtypes():
    a = Index([1, 2, 2, 3])
    b = Index(['1', '0', '0'])
    expected = Index([1, 2, 2, 3, '1', '0', '0'])
    result = a.union(b, sort=False)
    tm.assert_index_equal(result, expected)