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
class TestDataFramesWithMultiIndex(TestDataFrames):

    @classmethod
    def setup_class(cls):
        cls.endog = pd.DataFrame(np.random.random(10), columns=['y_1'])
        mi = pd.MultiIndex.from_product([['x'], ['1', '2']])
        exog = pd.DataFrame(np.random.random((10, 2)), columns=mi)
        exog_flattened_idx = pd.Index(['const', 'x_1', 'x_2'])
        exog.insert(0, 'const', 1)
        cls.exog = exog
        cls.data = sm_data.handle_data(cls.endog, cls.exog)
        nrows = 10
        nvars = 3
        cls.col_input = np.random.random(nvars)
        cls.col_result = pd.Series(cls.col_input, index=exog_flattened_idx)
        cls.row_input = np.random.random(nrows)
        cls.row_result = pd.Series(cls.row_input, index=exog.index)
        cls.cov_input = np.random.random((nvars, nvars))
        cls.cov_result = pd.DataFrame(cls.cov_input, index=exog_flattened_idx, columns=exog_flattened_idx)
        cls.xnames = ['const', 'x_1', 'x_2']
        cls.ynames = 'y_1'
        cls.row_labels = cls.exog.index