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
class CompareL11D(CompareL1):
    """
    Check t and f tests.  This only works for 1-d results
    """

    def test_tests(self):
        restrictmat = np.eye(len(self.res1.params.ravel()))
        assert_almost_equal(self.res1.t_test(restrictmat).pvalue, self.res2.t_test(restrictmat).pvalue, DECIMAL_4)
        assert_almost_equal(self.res1.f_test(restrictmat).pvalue, self.res2.f_test(restrictmat).pvalue, DECIMAL_4)