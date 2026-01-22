import os
import numpy as np
from numpy.testing import assert_almost_equal
import pandas as pd
from statsmodels.tsa.stattools import bds
class TestBDSGDPC1(CheckBDS):
    """
    BDS Test on GDPC1: 1947Q1 - 2013Q1

    References
    ----------
    http://research.stlouisfed.org/fred2/series/GDPC1
    """

    @classmethod
    def setup_class(cls):
        cls.results = results[results[0] == 4]
        cls.bds_stats = np.array(cls.results[2].iloc[1:])
        cls.pvalues = np.array(cls.results[3].iloc[1:])
        cls.data = data[3][data[3].notnull()]
        cls.res = bds(cls.data, 5)