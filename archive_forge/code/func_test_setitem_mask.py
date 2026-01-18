import collections
import operator
import sys
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.json.array import (
@pytest.mark.parametrize('mask', [np.array([True, True, True, False, False]), pd.array([True, True, True, False, False], dtype='boolean'), pd.array([True, True, True, pd.NA, pd.NA], dtype='boolean')], ids=['numpy-array', 'boolean-array', 'boolean-array-na'])
def test_setitem_mask(self, data, mask, box_in_series, request):
    if box_in_series:
        mark = pytest.mark.xfail(reason='cannot set using a list-like indexer with a different length')
        request.applymarker(mark)
    elif not isinstance(mask, np.ndarray):
        mark = pytest.mark.xfail(reason='Issues unwanted DeprecationWarning')
        request.applymarker(mark)
    super().test_setitem_mask(data, mask, box_in_series)