from statsmodels.compat.pandas import MONTH_END
from statsmodels.compat.pytest import pytest_warns
import os
import re
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
import scipy.stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tsa.holtwinters import (
from statsmodels.tsa.holtwinters._exponential_smoothers import (
from statsmodels.tsa.holtwinters._smoothers import (
@pytest.mark.parametrize('params', [[0.8, 0.3, 0.9], [0.3, 0.8, 0.2], [0.5, 0.6, 0.6]])
def test_to_restricted_equiv(params):
    params = np.array(params)
    sel = np.array([True] * 3)
    bounds = np.array([[0.0, 1.0]] * 3)
    assert_allclose(to_restricted(params, sel, bounds), _test_to_restricted(params, sel.astype(np.int64), bounds))