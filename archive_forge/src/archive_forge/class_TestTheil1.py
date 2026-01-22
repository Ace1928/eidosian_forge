import os
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from statsmodels.regression.linear_model import OLS, GLS
from statsmodels.sandbox.regression.penalized import TheilGLS
class TestTheil1(CheckEquivalenceMixin):

    @classmethod
    def setup_class(cls):
        y, x = cls.get_sample()
        mod1 = TheilGLS(y, x, sigma_prior=[0, 0, 1.0, 1.0])
        cls.res1 = mod1.fit(200000)
        cls.res2 = OLS(y, x[:, :3]).fit()