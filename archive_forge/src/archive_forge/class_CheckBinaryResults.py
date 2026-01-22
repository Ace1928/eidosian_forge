from statsmodels.compat.pandas import assert_index_equal
import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy import stats
from scipy.stats import nbinom
import statsmodels.api as sm
from statsmodels.discrete.discrete_margins import _iscount, _isdummy
from statsmodels.discrete.discrete_model import (
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import (
from .results.results_discrete import Anes, DiscreteL1, RandHIE, Spector
class CheckBinaryResults(CheckModelResults):

    def test_pred_table(self):
        assert_array_equal(self.res1.pred_table(), self.res2.pred_table)

    def test_resid_dev(self):
        assert_almost_equal(self.res1.resid_dev, self.res2.resid_dev, DECIMAL_4)

    def test_resid_generalized(self):
        assert_almost_equal(self.res1.resid_generalized, self.res2.resid_generalized, DECIMAL_4)

    @pytest.mark.smoke
    def test_resid_response(self):
        self.res1.resid_response