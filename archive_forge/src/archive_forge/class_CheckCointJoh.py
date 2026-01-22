import os
import warnings
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tsa.vector_ar.vecm import coint_johansen
class CheckCointJoh:

    def test_basic(self):
        assert_equal(self.res.ind, np.arange(len(self.res.ind), dtype=int))
        assert_equal(self.res.r0t.shape, (self.nobs_r, 8))

    def test_table_trace(self):
        table1 = np.column_stack((self.res.lr1, self.res.cvt))
        assert_almost_equal(table1, self.res1_m.reshape(table1.shape, order='F'))

    def test_table_maxeval(self):
        table2 = np.column_stack((self.res.lr2, self.res.cvm))
        assert_almost_equal(table2, self.res2_m.reshape(table2.shape, order='F'))

    def test_normalization(self):
        evec = self.res.evec
        non_zero = evec.flat != 0
        assert evec.flat[non_zero][0] > 0