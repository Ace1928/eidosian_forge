import os
import re
import sys
from contextlib import contextmanager
import numpy as np
import pytest
from numpy.testing import (
from scipy.linalg import norm
from scipy.optimize import fmin_bfgs
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.linear_model._theil_sen import (
from sklearn.utils._testing import assert_almost_equal
def test_theil_sen_1d_no_intercept():
    X, y, w, c = gen_toy_problem_1d(intercept=False)
    lstq = LinearRegression(fit_intercept=False).fit(X, y)
    assert np.abs(lstq.coef_ - w - c) > 0.5
    theil_sen = TheilSenRegressor(fit_intercept=False, random_state=0).fit(X, y)
    assert_array_almost_equal(theil_sen.coef_, w + c, 1)
    assert_almost_equal(theil_sen.intercept_, 0.0)
    theil_sen.score(X, y)