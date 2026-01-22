import os
import numpy as np
from numpy.testing import assert_almost_equal
import pandas as pd
from statsmodels.tsa.stattools import bds
class CheckBDS:
    """
    Test BDS

    Test values from Kanzler's MATLAB program bds.
    """

    def test_stat(self):
        assert_almost_equal(self.res[0], self.bds_stats, DECIMAL_8)

    def test_pvalue(self):
        assert_almost_equal(self.res[1], self.pvalues, DECIMAL_8)