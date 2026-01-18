from statsmodels.compat.pandas import PD_LT_2_2_0, YEAR_END, is_int_index
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_raises
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.base import tsa_model
def test_range_index():
    tsa_model.__warningregistry__ = {}
    endog = pd.Series(np.random.normal(size=5))
    assert_equal(isinstance(endog.index, pd.RangeIndex), True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        mod = tsa_model.TimeSeriesModel(endog)
        assert_equal(len(w), 0)