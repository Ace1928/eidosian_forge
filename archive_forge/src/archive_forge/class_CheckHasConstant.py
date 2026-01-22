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
class CheckHasConstant:

    def test_hasconst(self):
        for x, result in zip(self.exogs, self.results):
            mod = self.mod(self.y, x)
            assert_equal(mod.k_constant, result[0])
            assert_equal(mod.data.k_constant, result[0])
            if result[1] is None:
                assert_(mod.data.const_idx is None)
            else:
                assert_equal(mod.data.const_idx, result[1])
            fit_kwds = getattr(self, 'fit_kwds', {})
            try:
                res = mod.fit(**fit_kwds)
            except np.linalg.LinAlgError:
                pass
            else:
                assert_equal(res.model.k_constant, result[0])
                assert_equal(res.model.data.k_constant, result[0])

    @classmethod
    def setup_class(cls):
        np.random.seed(0)
        cls.y_c = np.random.randn(20)
        cls.y_bin = (cls.y_c > 0).astype(int)
        x1 = np.column_stack((np.ones(20), np.zeros(20)))
        result1 = (1, 0)
        x2 = np.column_stack((np.arange(20) < 10.5, np.arange(20) > 10.5)).astype(float)
        result2 = (1, None)
        x3 = np.column_stack((np.arange(20), np.zeros(20)))
        result3 = (0, None)
        x4 = np.column_stack((np.arange(20), np.zeros((20, 2))))
        result4 = (0, None)
        x5 = np.column_stack((np.zeros(20), 0.5 * np.ones(20)))
        result5 = (1, 1)
        x5b = np.column_stack((np.arange(20), np.ones((20, 3))))
        result5b = (1, 1)
        x5c = np.column_stack((np.arange(20), np.ones((20, 3)) * [0.5, 1, 1]))
        result5c = (1, 2)
        x6 = np.column_stack((np.arange(20) < 10.5, np.arange(20) > 10.5, np.zeros(20))).astype(float)
        result6 = (1, None)
        x7 = np.column_stack((np.arange(20) < 10.5, np.arange(20) > 10.5, np.zeros((20, 2)))).astype(float)
        result7 = (1, None)
        cls.exogs = (x1, x2, x3, x4, x5, x5b, x5c, x6, x7)
        cls.results = (result1, result2, result3, result4, result5, result5b, result5c, result6, result7)
        cls._initialize()