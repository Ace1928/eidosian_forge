from datetime import datetime
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.cast import find_common_type
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
@pytest.mark.parametrize('klass', [np.array, Series, list])
@pytest.mark.parametrize('index', ['string'], indirect=True)
def test_union_from_iterables(self, index, klass, sort):
    first = index[5:20]
    second = index[:10]
    everything = index[:20]
    case = klass(second.values)
    result = first.union(case, sort=sort)
    if sort in (None, False):
        tm.assert_index_equal(result.sort_values(), everything.sort_values())
    else:
        tm.assert_index_equal(result, everything)