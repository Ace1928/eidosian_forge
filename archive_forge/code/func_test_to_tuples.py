from itertools import permutations
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
@pytest.mark.parametrize('tuples', [list(zip(range(10), range(1, 11))), list(zip(date_range('20170101', periods=10), date_range('20170101', periods=10))), list(zip(timedelta_range('0 days', periods=10), timedelta_range('1 day', periods=10)))])
def test_to_tuples(self, tuples):
    idx = IntervalIndex.from_tuples(tuples)
    result = idx.to_tuples()
    expected = Index(com.asarray_tuplesafe(tuples))
    tm.assert_index_equal(result, expected)