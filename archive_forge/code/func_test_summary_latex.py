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
def test_summary_latex(self):
    summ = self.res1.summary()
    ltx = summ.as_latex()
    n_lines = len(ltx.splitlines())
    if not isinstance(self.res1.model, MNLogit):
        assert n_lines == 19 + np.size(self.res1.params)
    assert 'Covariance Type:' in ltx