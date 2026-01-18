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
def test_spatial_median_1d():
    X = np.array([1.0, 2.0, 3.0]).reshape(3, 1)
    true_median = 2.0
    _, median = _spatial_median(X)
    assert_array_almost_equal(median, true_median)
    random_state = np.random.RandomState(0)
    X = random_state.randint(100, size=(1000, 1))
    true_median = np.median(X.ravel())
    _, median = _spatial_median(X)
    assert_array_equal(median, true_median)