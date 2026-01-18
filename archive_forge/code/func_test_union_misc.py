import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_union_misc(self, sort):
    index = period_range('1/1/2000', '1/20/2000', freq='D')
    result = index[:-5].union(index[10:], sort=sort)
    tm.assert_index_equal(result, index)
    result = _permute(index[:-5]).union(_permute(index[10:]), sort=sort)
    if sort is False:
        tm.assert_index_equal(result.sort_values(), index)
    else:
        tm.assert_index_equal(result, index)
    index = period_range('1/1/2000', '1/20/2000', freq='D')
    index2 = period_range('1/1/2000', '1/20/2000', freq='W-WED')
    result = index.union(index2, sort=sort)
    expected = index.astype(object).union(index2.astype(object), sort=sort)
    tm.assert_index_equal(result, expected)