from statsmodels.compat.pandas import MONTH_END
import warnings
import numpy as np
from numpy.testing import assert_, assert_allclose
import pandas as pd
import pytest
from scipy.stats import ortho_group
from statsmodels.tools.sm_exceptions import EstimationWarning
from statsmodels.tsa.statespace import (
from statsmodels.tsa.vector_ar.tests.test_var import get_macrodata
def test_time_varying_ssm():
    mod = sarimax.SARIMAX([0] * 11, order=(1, 0, 0))
    mod.update([0.5, 1.0])
    T = np.zeros((1, 1, 11))
    T[..., :5] = 0.5
    T[..., 5:] = 0.2
    mod['transition'] = T
    irfs = mod.ssm.impulse_responses()
    desired = np.cumprod(np.r_[1, [0.5] * 4, [0.2] * 5]).reshape(10, 1)
    assert_allclose(irfs, desired)