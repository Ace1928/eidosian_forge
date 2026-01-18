import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('p_values, o_values, values, expected_values', [([Period('2019Q1', 'Q-DEC'), Period('2019Q2', 'Q-DEC')], [Period('2019Q1', 'Q-DEC'), Period('2019Q2', 'Q-DEC'), 'All'], [1.0, 1.0], [1.0, 1.0, np.nan]), ([Period('2019Q1', 'Q-DEC'), Period('2019Q2', 'Q-DEC')], [Period('2019Q1', 'Q-DEC'), Period('2019Q2', 'Q-DEC')], [1.0, 1.0], [1.0, 1.0])])
def test_reindex_periodindex_with_object(p_values, o_values, values, expected_values):
    period_index = PeriodIndex(p_values)
    object_index = Index(o_values)
    ser = Series(values, index=period_index)
    result = ser.reindex(object_index)
    expected = Series(expected_values, index=object_index)
    tm.assert_series_equal(result, expected)