from statsmodels.compat.pandas import PD_LT_2_2_0, YEAR_END, is_int_index
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_raises
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.base import tsa_model
def test_prediction_increment_unsupported():
    endog = dta[2].copy()
    endog.index = unsupported_indexes[-2][0]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('ignore')
        mod = tsa_model.TimeSeriesModel(endog)
    start_key = 0
    end_key = None
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 0)
    assert_equal(end, nobs - 1)
    assert_equal(out_of_sample, 0)
    assert_equal(prediction_index.equals(mod.data.row_labels), True)
    start_key = -2
    end_key = -1
    start, end, out_of_sample, prediction_index = mod._get_prediction_index(start_key, end_key)
    assert_equal(start, 3)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 0)
    assert_equal(prediction_index.equals(mod.data.row_labels[3:]), True)
    start_key = 1
    end_key = nobs
    message = 'No supported index is available. Prediction results will be given with an integer index beginning at `start`.'
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        start, end, out_of_sample, prediction_index = mod._get_prediction_index(start_key, end_key)
        assert_equal(str(w[0].message), message)
    assert_equal(start, 1)
    assert_equal(end, 4)
    assert_equal(out_of_sample, 1)
    assert_equal(prediction_index.equals(pd.Index(np.arange(1, 6))), True)
    loc, index, index_was_expanded = mod._get_index_loc(2)
    assert_equal(loc, 2)
    desired_index = pd.RangeIndex(start=0, stop=3, step=1)
    assert_equal(index.equals(desired_index), True)
    assert_equal(index_was_expanded, False)
    loc, index, index_was_expanded = mod._get_index_label_loc(2)
    assert_equal(loc, 2)
    desired_index = pd.RangeIndex(start=0, stop=3, step=1)
    assert_equal(index.equals(desired_index), True)
    assert_equal(index_was_expanded, False)
    loc, index, index_was_expanded = mod._get_index_label_loc('c')
    assert_equal(loc, 2)
    desired_index = mod.data.row_labels[:3]
    assert_equal(index.equals(desired_index), True)
    assert_equal(index_was_expanded, False)