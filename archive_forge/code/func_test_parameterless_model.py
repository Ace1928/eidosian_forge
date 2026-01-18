import numpy as np
import pandas as pd
import os
import pytest
from numpy.testing import assert_, assert_equal, assert_allclose
from statsmodels.tsa.statespace.exponential_smoothing import (
def test_parameterless_model(reset_randomstate):
    x = np.cumsum(np.random.standard_normal(1000))
    ses = ExponentialSmoothing(x, initial_level=x[0], initialization_method='known')
    with ses.fix_params({'smoothing_level': 0.5}):
        res = ses.fit()
    assert np.isnan(res.bse).all()
    assert res.fixed_params == ['smoothing_level']