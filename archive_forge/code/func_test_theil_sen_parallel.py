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
def test_theil_sen_parallel():
    X, y, w, c = gen_toy_problem_2d()
    lstq = LinearRegression().fit(X, y)
    assert norm(lstq.coef_ - w) > 1.0
    theil_sen = TheilSenRegressor(n_jobs=2, random_state=0, max_subpopulation=2000.0).fit(X, y)
    assert_array_almost_equal(theil_sen.coef_, w, 1)
    assert_array_almost_equal(theil_sen.intercept_, c, 1)