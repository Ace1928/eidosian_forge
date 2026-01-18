import os
import warnings
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tsa.vector_ar.vecm import coint_johansen
def test_table_trace(self):
    table1 = np.column_stack((self.res.lr1, self.res.cvt))
    assert_almost_equal(table1, self.res1_m.reshape(table1.shape, order='F'))