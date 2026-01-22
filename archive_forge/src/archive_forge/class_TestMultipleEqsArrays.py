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
class TestMultipleEqsArrays(TestArrays):

    @classmethod
    def setup_class(cls):
        cls.endog = np.random.random((10, 4))
        cls.exog = np.c_[np.ones(10), np.random.random((10, 2))]
        cls.data = sm_data.handle_data(cls.endog, cls.exog)
        nrows = 10
        nvars = 3
        neqs = 4
        cls.col_result = cls.col_input = np.random.random(nvars)
        cls.row_result = cls.row_input = np.random.random(nrows)
        cls.cov_result = cls.cov_input = np.random.random((nvars, nvars))
        cls.cov_eq_result = cls.cov_eq_input = np.random.random((neqs, neqs))
        cls.col_eq_result = cls.col_eq_input = np.array((neqs, nvars))
        cls.xnames = ['const', 'x1', 'x2']
        cls.ynames = ['y1', 'y2', 'y3', 'y4']
        cls.row_labels = None

    def test_attach(self):
        data = self.data
        np.testing.assert_equal(data.wrap_output(self.col_input, 'columns'), self.col_result)
        np.testing.assert_equal(data.wrap_output(self.row_input, 'rows'), self.row_result)
        np.testing.assert_equal(data.wrap_output(self.cov_input, 'cov'), self.cov_result)
        np.testing.assert_equal(data.wrap_output(self.cov_eq_input, 'cov_eq'), self.cov_eq_result)
        np.testing.assert_equal(data.wrap_output(self.col_eq_input, 'columns_eq'), self.col_eq_result)