from statsmodels.compat.pandas import PD_LT_2_2_0, YEAR_END, is_int_index
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_raises
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.base import tsa_model
def test_prediction_increment_pandas_dates_nanosecond():
    endog = dta[2].copy()
    endog.index = pd.date_range(start='1970-01-01', periods=len(endog), freq='ns')
    mod = tsa_model.TimeSeriesModel(endog)
    start_key = 0
    end_key = None
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 0)
    assert_equal(end, nobs - 1)
    assert_equal(out_of_sample, 0)
    assert type(prediction_index) is type(endog.index)
    assert_equal(prediction_index.equals(mod._index), True)
    start_key = -2
    end_key = -1
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 3)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 0)
    assert type(prediction_index) is type(endog.index)
    assert_equal(prediction_index.equals(mod._index[3:]), True)
    start_key = 1
    end_key = nobs
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 1)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 1)
    desired_index = pd.date_range(start='1970-01-01', periods=6, freq='ns')[1:]
    assert_equal(prediction_index.equals(desired_index), True)
    start_key = pd.Timestamp('1970-01-01')
    end_key = pd.Timestamp(start_key.value + 7)
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 0)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 3)
    desired_index = pd.date_range(start='1970-01-01', periods=8, freq='ns')
    assert_equal(prediction_index.equals(desired_index), True)