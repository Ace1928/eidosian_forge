from statsmodels.compat.pandas import PD_LT_2_2_0, YEAR_END, is_int_index
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_raises
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.base import tsa_model
@pytest.mark.xfail(reason='Pandas PeriodIndex.is_full does not yet work for all frequencies (e.g. frequencies with a multiplier, like "2Q").')
def test_nonfull_periodindex():
    index = pd.PeriodIndex(['2000-01', '2000-03'], freq='M')
    endog = pd.Series(np.zeros(len(index)), index=index)
    message = 'A Period index has been provided, but it is not full and so will be ignored when e.g. forecasting.'
    with pytest.warns(ValueWarning, match=message):
        tsa_model.TimeSeriesModel(endog)