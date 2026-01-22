import os
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from statsmodels.regression.linear_model import OLS, GLS
from statsmodels.sandbox.regression.penalized import TheilGLS
class TestTheilLinRestriction(CheckEquivalenceMixin):

    @classmethod
    def setup_class(cls):
        y, x = cls.get_sample()
        x2 = x[:, :2].copy()
        x2[:, 1] += x[:, 2]
        mod1 = TheilGLS(y, x[:, :3], r_matrix=[[0, 1, -1]])
        cls.res1 = mod1.fit(200000)
        cls.res2 = OLS(y, x2).fit()
        tol = {'pvalues': (0.0001, 2e-07), 'tvalues': (0.0005, 0)}
        tol.update(cls.tol)
        cls.tol = tol