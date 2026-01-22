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
def _simple_dbl_exp_smoother(x, alpha, beta, l0, b0, nforecast=0):
    """
    Simple, slow, direct implementation of double exp smoothing for testing
    """
    n = x.shape[0]
    lvals = np.zeros(n)
    b = np.zeros(n)
    xhat = np.zeros(n)
    f = np.zeros(nforecast)
    lvals[0] = l0
    b[0] = b0
    xhat[0] = l0 + b0
    lvals[0] = alpha * x[0] + (1 - alpha) * (l0 + b0)
    b[0] = beta * (lvals[0] - l0) + (1 - beta) * b0
    for t in range(1, n):
        lvals[t] = alpha * x[t] + (1 - alpha) * (lvals[t - 1] + b[t - 1])
        b[t] = beta * (lvals[t] - lvals[t - 1]) + (1 - beta) * b[t - 1]
    xhat[1:] = lvals[0:-1] + b[0:-1]
    f[:] = lvals[-1] + np.arange(1, nforecast + 1) * b[-1]
    err = x - xhat
    return (lvals, b, f, err, xhat)