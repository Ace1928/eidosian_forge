import os
import warnings
import numpy as np
import statsmodels.stats.contingency_tables as ctab
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
def test_shifting():
    t = np.zeros((3, 4), dtype=np.float64)
    result = np.full((3, 4), 0.5)
    assert_equal(ctab.Table(t, shift_zeros=False).table, t)
    assert_equal(ctab.Table(t, shift_zeros=True).table, result)
    t = np.asarray([[0, 1, 2], [3, 0, 4], [5, 6, 0]], dtype=np.float64)
    r = np.asarray([[0.5, 1, 2], [3, 0.5, 4], [5, 6, 0.5]], dtype=np.float64)
    assert_equal(ctab.Table(t).table, r)
    assert_equal(ctab.Table(t, shift_zeros=True).table, r)