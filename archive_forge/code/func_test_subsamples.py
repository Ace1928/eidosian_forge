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
def test_subsamples():
    X, y, w, c = gen_toy_problem_4d()
    theil_sen = TheilSenRegressor(n_subsamples=X.shape[0], random_state=0).fit(X, y)
    lstq = LinearRegression().fit(X, y)
    assert_array_almost_equal(theil_sen.coef_, lstq.coef_, 9)