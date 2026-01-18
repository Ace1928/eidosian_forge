import os
import warnings
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tsa.vector_ar.vecm import coint_johansen
def test_table_maxeval(self):
    table2 = np.column_stack((self.res.lr2, self.res.cvm))
    assert_almost_equal(table2, self.res2_m.reshape(table2.shape, order='F'))