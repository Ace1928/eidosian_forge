from statsmodels.compat.pandas import PD_LT_2_2_0
from datetime import datetime
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tools.testing import assert_equal
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
def test_period_index():
    dates = pd.period_range(start='1/1/1990', periods=20, freq='M')
    x = np.arange(1, 21.0)
    model = TimeSeriesModel(pd.Series(x, index=dates))
    assert_equal(model._index.freqstr, 'M')
    model = TimeSeriesModel(pd.Series(x, index=dates))
    npt.assert_(model.data.freq == 'M')