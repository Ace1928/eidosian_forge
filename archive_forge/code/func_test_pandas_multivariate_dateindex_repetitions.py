from statsmodels.compat.pandas import MONTH_END
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from scipy.signal import lfilter
from statsmodels.tools.sm_exceptions import (
from statsmodels.tsa.statespace import (
from .test_impulse_responses import TVSS
def test_pandas_multivariate_dateindex_repetitions():
    ix = pd.date_range(start='2000', periods=2, freq=MONTH_END)
    endog = pd.DataFrame(np.zeros((2, 2)), columns=['y1', 'y2'], index=ix)
    mod = varmax.VARMAX(endog, trend='n')
    res = mod.filter([0.5, 0.0, 0.0, 0.2, 1.0, 0.0, 1.0])
    actual = res.simulate(2, state_shocks=np.zeros((2, 2)), initial_state=np.zeros(2), repetitions=2)
    columns = pd.MultiIndex.from_product([['y1', 'y2'], [0, 1]])
    desired = pd.DataFrame(np.zeros((2, 4)), columns=columns, index=ix)
    assert_allclose(actual, desired)
    assert_(actual.columns.equals(desired.columns))
    actual = res.simulate(2, anchor=2, state_shocks=np.zeros((2, 2)), initial_state=np.zeros(2), repetitions=2)
    ix = pd.date_range(start='2000-03', periods=2, freq=MONTH_END)
    columns = pd.MultiIndex.from_product([['y1', 'y2'], [0, 1]])
    desired = pd.DataFrame(np.zeros((2, 4)), index=ix, columns=columns)
    assert_allclose(actual, desired)
    assert_(actual.index.equals(desired.index))
    assert_(actual.columns.equals(desired.columns))