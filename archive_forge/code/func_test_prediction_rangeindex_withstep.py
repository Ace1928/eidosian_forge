from statsmodels.compat.pandas import PD_LT_2_2_0, YEAR_END, is_int_index
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_raises
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.base import tsa_model
def test_prediction_rangeindex_withstep():
    index = supported_increment_indexes[3][0]
    endog = pd.Series(dta[0], index=index)
    mod = tsa_model.TimeSeriesModel(endog)
    start_key = 0
    end_key = None
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 0)
    assert_equal(end, nobs - 1)
    assert_equal(out_of_sample, 0)
    desired_index = pd.RangeIndex(start=0, stop=nobs * 6, step=6)
    assert_equal(prediction_index.equals(desired_index), True)
    start_key = -2
    end_key = -1
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 3)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 0)
    desired_index = pd.RangeIndex(start=3 * 6, stop=nobs * 6, step=6)
    assert_equal(prediction_index.equals(desired_index), True)
    start_key = 1
    end_key = nobs
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 1)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 1)
    desired_index = pd.RangeIndex(start=1 * 6, stop=(nobs + 1) * 6, step=6)
    assert_equal(prediction_index.equals(desired_index), True)
    loc, index, index_was_expanded = mod._get_index_loc(2)
    assert_equal(loc, 2)
    desired_index = pd.RangeIndex(start=0, stop=3 * 6, step=6)
    assert_equal(index.equals(desired_index), True)
    assert_equal(index_was_expanded, False)