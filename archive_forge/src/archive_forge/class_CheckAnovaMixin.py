from statsmodels.compat.pytest import pytest_warns
from statsmodels.compat.pandas import assert_index_equal, assert_series_equal
from statsmodels.compat.platform import (
from statsmodels.compat.scipy import SCIPY_GT_14
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.formula.api import glm, ols
import statsmodels.tools._testing as smt
from statsmodels.tools.sm_exceptions import HessianInversionWarning
class CheckAnovaMixin:

    @classmethod
    def setup_class(cls):
        import statsmodels.stats.tests.test_anova as ttmod
        test = ttmod.TestAnova3()
        test.setup_class()
        cls.data = test.data.drop([0, 1, 2])
        cls.initialize()

    def test_combined(self):
        res = self.res
        wa = res.wald_test_terms(skip_single=False, combine_terms=['Duration', 'Weight'], scalar=True)
        eye = np.eye(len(res.params))
        c_const = eye[0]
        c_w = eye[[2, 3]]
        c_d = eye[1]
        c_dw = eye[[4, 5]]
        c_weight = eye[2:6]
        c_duration = eye[[1, 4, 5]]
        compare_waldres(res, wa, [c_const, c_d, c_w, c_dw, c_duration, c_weight])

    def test_categories(self):
        res = self.res
        wa = res.wald_test_terms(skip_single=True, scalar=True)
        eye = np.eye(len(res.params))
        c_w = eye[[2, 3]]
        c_dw = eye[[4, 5]]
        compare_waldres(res, wa, [c_w, c_dw])