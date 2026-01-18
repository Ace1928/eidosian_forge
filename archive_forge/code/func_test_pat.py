import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.imputation.bayes_mi import BayesGaussMI, MI
from numpy.testing import assert_allclose, assert_equal
def test_pat():
    x = np.asarray([[1, np.nan, 3], [np.nan, 2, np.nan], [3, np.nan, 0], [np.nan, 1, np.nan], [3, 2, 1]])
    bm = BayesGaussMI(x)
    assert_allclose(bm.patterns[0], np.r_[0, 2])
    assert_allclose(bm.patterns[1], np.r_[1, 3])