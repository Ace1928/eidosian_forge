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
class TestWLSScalarVsArray(CheckRegressionResults):

    @classmethod
    def setup_class(cls):
        from statsmodels.datasets.longley import load
        dta = load()
        endog = np.asarray(dta.endog)
        exog = np.asarray(dta.exog)
        exog = add_constant(exog, prepend=True)
        wls_scalar = WLS(endog, exog, weights=1.0 / 3).fit()
        weights = [1 / 3.0] * len(endog)
        wls_array = WLS(endog, exog, weights=weights).fit()
        cls.res1 = wls_scalar
        cls.res2 = wls_array