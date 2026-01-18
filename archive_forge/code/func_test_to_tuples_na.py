from itertools import permutations
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
@pytest.mark.parametrize('tuples', [list(zip(range(10), range(1, 11))) + [np.nan], list(zip(date_range('20170101', periods=10), date_range('20170101', periods=10))) + [np.nan], list(zip(timedelta_range('0 days', periods=10), timedelta_range('1 day', periods=10))) + [np.nan]])
@pytest.mark.parametrize('na_tuple', [True, False])
def test_to_tuples_na(self, tuples, na_tuple):
    idx = IntervalIndex.from_tuples(tuples)
    result = idx.to_tuples(na_tuple=na_tuple)
    expected_notna = Index(com.asarray_tuplesafe(tuples[:-1]))
    result_notna = result[:-1]
    tm.assert_index_equal(result_notna, expected_notna)
    result_na = result[-1]
    if na_tuple:
        assert isinstance(result_na, tuple)
        assert len(result_na) == 2
        assert all((isna(x) for x in result_na))
    else:
        assert isna(result_na)