from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
@pytest.mark.parametrize('method,expected,sort', [('intersection', np.array([(1, 'A'), (2, 'A'), (1, 'B'), (2, 'B')], dtype=[('num', int), ('let', 'S1')]), False), ('intersection', np.array([(1, 'A'), (1, 'B'), (2, 'A'), (2, 'B')], dtype=[('num', int), ('let', 'S1')]), None), ('union', np.array([(1, 'A'), (1, 'B'), (1, 'C'), (2, 'A'), (2, 'B'), (2, 'C')], dtype=[('num', int), ('let', 'S1')]), None)])
def test_tuple_union_bug(self, method, expected, sort):
    index1 = Index(np.array([(1, 'A'), (2, 'A'), (1, 'B'), (2, 'B')], dtype=[('num', int), ('let', 'S1')]))
    index2 = Index(np.array([(1, 'A'), (2, 'A'), (1, 'B'), (2, 'B'), (1, 'C'), (2, 'C')], dtype=[('num', int), ('let', 'S1')]))
    result = getattr(index1, method)(index2, sort=sort)
    assert result.ndim == 1
    expected = Index(expected)
    tm.assert_index_equal(result, expected)