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
def test_valid_bounds(ses):
    bounds = {'smoothing_level': (0.1, 1.0)}
    res = ExponentialSmoothing(ses, bounds=bounds, initialization_method='estimated').fit(method='least_squares')
    res2 = ExponentialSmoothing(ses, initialization_method='estimated').fit(method='least_squares')
    assert_allclose(res.params['smoothing_level'], res2.params['smoothing_level'], rtol=0.0001)