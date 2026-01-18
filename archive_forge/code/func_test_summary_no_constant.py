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
def test_summary_no_constant():
    rs = np.random.RandomState(0)
    x = rs.standard_normal((100, 2))
    y = rs.standard_normal(100)
    summary = OLS(y, x).fit().summary()
    assert 'R² is computed ' in summary.as_text()