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
@pytest.mark.parametrize('idx', [('mid',), ('mid', 'btm'), ('mid', 'btm', 'top'), ('mid',), ('mid', 'top'), ('mid', 'top', 'btm'), ('btm',), ('btm', 'mid'), ('btm', 'mid', 'top'), ('btm',), ('btm', 'top'), ('btm', 'top', 'mid'), ('top',), ('top', 'mid'), ('top', 'mid', 'btm'), ('top',), ('top', 'btm'), ('top', 'btm', 'mid')])
def test_reindex_level_verify_first_level_repeats(self, idx):
    df = DataFrame({'jim': ['mid'] * 5 + ['btm'] * 8 + ['top'] * 7, 'joe': ['3rd'] * 2 + ['1st'] * 3 + ['2nd'] * 3 + ['1st'] * 2 + ['3rd'] * 3 + ['1st'] * 2 + ['3rd'] * 3 + ['2nd'] * 2, 'jolie': np.concatenate([np.random.default_rng(2).choice(1000, x, replace=False) for x in [2, 3, 3, 2, 3, 2, 3, 2]]), 'joline': np.random.default_rng(2).standard_normal(20).round(3) * 10})
    icol = ['jim', 'joe', 'jolie']

    def f(val):
        return np.nonzero((df['jim'] == val).to_numpy())[0]
    i = np.concatenate(list(map(f, idx)))
    left = df.set_index(icol).reindex(idx, level='jim')
    right = df.iloc[i].set_index(icol)
    tm.assert_frame_equal(left, right)