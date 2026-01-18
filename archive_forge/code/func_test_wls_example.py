from statsmodels.compat.python import lrange
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy.linalg import toeplitz
from scipy.stats import t as student_t
from statsmodels.datasets import longley
from statsmodels.regression.linear_model import (
from statsmodels.tools.tools import add_constant
def test_wls_example():
    Y = [1, 3, 4, 5, 2, 3, 4]
    x = lrange(1, 8)
    x = add_constant(x, prepend=False)
    wls_model = WLS(Y, x, weights=lrange(1, 8)).fit()
    assert_almost_equal(wls_model.fvalue, 0.127337843215, 6)
    assert_almost_equal(wls_model.scale, 2.44608530786 ** 2, 6)