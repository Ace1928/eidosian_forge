from datetime import (
import inspect
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
@pytest.mark.parametrize('idx, check_index_type', [[['C', 'B', 'A'], True], [['F', 'C', 'A', 'D'], True], [['A'], True], [['A', 'B', 'C'], True], [['C', 'A', 'B'], True], [['C', 'B'], True], [['C', 'A'], True], [['A', 'B'], True], [['B', 'A', 'C'], True], [['D', 'F'], False], [['A', 'C', 'B'], False]])
def test_reindex_level_verify_first_level(self, idx, check_index_type):
    df = DataFrame({'jim': list('B' * 4 + 'A' * 2 + 'C' * 3), 'joe': list('abcdeabcd')[::-1], 'jolie': [10, 20, 30] * 3, 'joline': np.random.default_rng(2).integers(0, 1000, 9)})
    icol = ['jim', 'joe', 'jolie']

    def f(val):
        return np.nonzero((df['jim'] == val).to_numpy())[0]
    i = np.concatenate(list(map(f, idx)))
    left = df.set_index(icol).reindex(idx, level='jim')
    right = df.iloc[i].set_index(icol)
    tm.assert_frame_equal(left, right, check_index_type=check_index_type)