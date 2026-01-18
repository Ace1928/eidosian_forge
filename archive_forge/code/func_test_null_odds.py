import os
import warnings
import numpy as np
import statsmodels.stats.contingency_tables as ctab
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
def test_null_odds(self):
    rslt = self.rslt.test_null_odds(correction=True)
    assert_allclose(rslt.statistic, self.mh_stat, rtol=0.0001, atol=1e-05)
    assert_allclose(rslt.pvalue, self.mh_pvalue, rtol=0.0001, atol=0.0001)