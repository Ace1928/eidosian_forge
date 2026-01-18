from statsmodels.compat.pandas import MONTH_END
import os
import re
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from statsmodels.datasets import nile
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.mlemodel import MLEModel, MLEResultsWrapper
from statsmodels.tsa.statespace.tests.results import (
def test_pandas_endog():
    endog = pd.Series([1.0, 2.0])
    warnings.simplefilter('always')
    dates = pd.date_range(start='1980-01-01', end='1981-01-01', freq='YS')
    endog = pd.Series([1.0, 2.0], index=dates)
    mod = check_endog(endog, **kwargs)
    mod.filter([])
    endog = pd.Series(['a', 'b'], index=dates)
    assert_raises(ValueError, check_endog, endog, **kwargs)
    endog = pd.Series([1.0, 2.0], index=dates)
    mod = check_endog(endog, **kwargs)
    mod.filter([])
    endog = pd.DataFrame({'a': [1.0, 2.0]}, index=dates)
    mod = check_endog(endog, **kwargs)
    mod.filter([])
    endog = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]}, index=dates)
    assert_raises(ValueError, check_endog, endog, **kwargs)
    endog = pd.DataFrame({'a': [1.0, 2.0]}, index=dates)
    mod = check_endog(endog, **kwargs)
    assert_equal(mod.endog.base is not mod.data.orig_endog, True)
    assert_equal(mod.endog.base is not endog, True)
    assert_equal(mod.data.orig_endog.values.base is not endog, True)
    endog.iloc[0, 0] = 2
    assert_equal(mod.endog, np.r_[1, 2].reshape(2, 1))
    assert_allclose(mod.data.orig_endog, endog)
    kwargs2 = {'k_states': 1, 'design': [[1], [0.0]], 'obs_cov': [[1, 0], [0, 1]], 'transition': [[1]], 'selection': [[1]], 'state_cov': [[1]], 'initialization': 'approximate_diffuse'}
    endog = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]}, index=dates)
    mod = check_endog(endog, k_endog=2, **kwargs2)
    mod.filter([])