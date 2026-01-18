from statsmodels.compat.pandas import PD_LT_2_2_0
from datetime import datetime
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tools.testing import assert_equal
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
def test_pandas_dates():
    data = [988, 819, 964]
    dates = ['2016-01-01 12:00:00', '2016-02-01 12:00:00', '2016-03-01 12:00:00']
    datetime_dates = pd.to_datetime(dates)
    result = pd.Series(data=data, index=datetime_dates, name='price')
    df = pd.DataFrame(data={'price': data}, index=pd.DatetimeIndex(dates, freq='MS'))
    model = TimeSeriesModel(df['price'])
    assert_equal(model.data.dates, result.index)