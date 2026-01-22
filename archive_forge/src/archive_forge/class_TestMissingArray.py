from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_equal, assert_, assert_raises
import pandas as pd
import pytest
from statsmodels.base import data as sm_data
from statsmodels.formula import handle_formula_data
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.discrete.discrete_model import Logit
class TestMissingArray:

    @classmethod
    def setup_class(cls):
        X = np.random.random((25, 4))
        y = np.random.random(25)
        y[10] = np.nan
        X[2, 3] = np.nan
        X[14, 2] = np.nan
        cls.y, cls.X = (y, X)

    @pytest.mark.smoke
    def test_raise_no_missing(self):
        sm_data.handle_data(np.random.random(20), np.random.random((20, 2)), 'raise')

    def test_raise(self):
        with pytest.raises(Exception):
            sm_data.handle_data(self.y, self.X, 'raise')

    def test_drop(self):
        y = self.y
        X = self.X
        combined = np.c_[y, X]
        idx = ~np.isnan(combined).any(axis=1)
        y = y[idx]
        X = X[idx]
        data = sm_data.handle_data(self.y, self.X, 'drop')
        np.testing.assert_array_equal(data.endog, y)
        np.testing.assert_array_equal(data.exog, X)

    def test_none(self):
        data = sm_data.handle_data(self.y, self.X, 'none', hasconst=False)
        np.testing.assert_array_equal(data.endog, self.y)
        np.testing.assert_array_equal(data.exog, self.X)
        assert data.k_constant == 0

    def test_endog_only_raise(self):
        with pytest.raises(Exception):
            sm_data.handle_data(self.y, None, 'raise')

    def test_endog_only_drop(self):
        y = self.y
        y = y[~np.isnan(y)]
        data = sm_data.handle_data(self.y, None, 'drop')
        np.testing.assert_array_equal(data.endog, y)

    def test_mv_endog(self):
        y = self.X
        y = y[~np.isnan(y).any(axis=1)]
        data = sm_data.handle_data(self.X, None, 'drop')
        np.testing.assert_array_equal(data.endog, y)

    def test_extra_kwargs_2d(self):
        sigma = np.random.random((25, 25))
        sigma = sigma + sigma.T - np.diag(np.diag(sigma))
        data = sm_data.handle_data(self.y, self.X, 'drop', sigma=sigma)
        idx = ~np.isnan(np.c_[self.y, self.X]).any(axis=1)
        sigma = sigma[idx][:, idx]
        np.testing.assert_array_equal(data.sigma, sigma)

    def test_extra_kwargs_1d(self):
        weights = np.random.random(25)
        data = sm_data.handle_data(self.y, self.X, 'drop', weights=weights)
        idx = ~np.isnan(np.c_[self.y, self.X]).any(axis=1)
        weights = weights[idx]
        np.testing.assert_array_equal(data.weights, weights)