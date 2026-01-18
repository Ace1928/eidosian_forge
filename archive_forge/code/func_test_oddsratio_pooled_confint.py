import os
import warnings
import numpy as np
import statsmodels.stats.contingency_tables as ctab
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
def test_oddsratio_pooled_confint(self):
    lcb, ucb = self.rslt.oddsratio_pooled_confint()
    assert_allclose(lcb, self.or_lcb, rtol=0.0001, atol=0.0001)
    assert_allclose(ucb, self.or_ucb, rtol=0.0001, atol=0.0001)