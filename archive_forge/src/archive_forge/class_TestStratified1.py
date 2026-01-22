import os
import warnings
import numpy as np
import statsmodels.stats.contingency_tables as ctab
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
class TestStratified1(CheckStratifiedMixin):
    """
    data = array(c(0, 0, 6, 5,
                   3, 0, 3, 6,
                   6, 2, 0, 4,
                   5, 6, 1, 0,
                   2, 5, 0, 0),
                   dim=c(2, 2, 5))
    rslt = mantelhaen.test(data)
    """

    @classmethod
    def setup_class(cls):
        tables = [None] * 5
        tables[0] = np.array([[0, 0], [6, 5]])
        tables[1] = np.array([[3, 0], [3, 6]])
        tables[2] = np.array([[6, 2], [0, 4]])
        tables[3] = np.array([[5, 6], [1, 0]])
        tables[4] = np.array([[2, 5], [0, 0]])
        cls.initialize(tables)
        cls.oddsratio_pooled = 7
        cls.logodds_pooled = np.log(7)
        cls.mh_stat = 3.9286
        cls.mh_pvalue = 0.04747
        cls.or_lcb = 1.026713
        cls.or_ucb = 47.725133