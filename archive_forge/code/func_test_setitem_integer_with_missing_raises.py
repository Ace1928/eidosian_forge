import collections
import operator
import sys
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.json.array import (
@pytest.mark.xfail(reason='list indices must be integers or slices, not NAType')
@pytest.mark.parametrize('idx, box_in_series', [([0, 1, 2, pd.NA], False), pytest.param([0, 1, 2, pd.NA], True, marks=pytest.mark.xfail(reason='GH-31948')), (pd.array([0, 1, 2, pd.NA], dtype='Int64'), False), (pd.array([0, 1, 2, pd.NA], dtype='Int64'), False)], ids=['list-False', 'list-True', 'integer-array-False', 'integer-array-True'])
def test_setitem_integer_with_missing_raises(self, data, idx, box_in_series):
    super().test_setitem_integer_with_missing_raises(data, idx, box_in_series)