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
def test_norm_resid(self):
    resid = self.res1.wresid
    norm_resid = resid / np.sqrt(np.sum(resid ** 2.0) / self.res1.df_resid)
    model_norm_resid = self.res1.resid_pearson
    assert_almost_equal(model_norm_resid, norm_resid, DECIMAL_7)