import os
import warnings
import numpy as np
import statsmodels.stats.contingency_tables as ctab
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
def test_riskratio_pvalue(self):
    assert_allclose(self.tbl_obj.riskratio_pvalue(), self.riskratio_pvalue)