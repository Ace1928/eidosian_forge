from statsmodels.compat.platform import PLATFORM_OSX
import os
import csv
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
import pytest
from statsmodels.regression.mixed_linear_model import (
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from statsmodels.base import _penalties as penalties
import statsmodels.tools.numdiff as nd
from .results import lme_r_results
def test_summary_col():
    from statsmodels.iolib.summary2 import summary_col
    ids = [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    y = np.array([1.727, -1.037, 2.904, 3.569, 4.629, 5.736, 6.747, 7.02, 5.624, 10.155, 10.4, 17.164, 17.276, 14.988, 14.453])
    d = {'Y': y, 'X': x, 'IDS': ids}
    d = pd.DataFrame(d)
    sp1 = np.array([-1.26722599, 1.1617587, 0.19547518])
    mod1 = MixedLM.from_formula('Y ~ X', d, groups=d['IDS'])
    results1 = mod1.fit(start_params=sp1)
    sp2 = np.array([3.48416861, 0.55287862, 1.38537901])
    mod2 = MixedLM.from_formula('X ~ Y', d, groups=d['IDS'])
    results2 = mod2.fit(start_params=sp2)
    out = summary_col([results1, results2], stars=True, regressor_order=['Group Var', 'Intercept', 'X', 'Y'])
    s = '\n=============================\n              Y         X    \n-----------------------------\nGroup Var 0.1955    1.3854   \n          (0.6032)  (2.7377) \nIntercept -1.2672   3.4842*  \n          (1.6546)  (1.8882) \nX         1.1618***          \n          (0.1959)           \nY                   0.5529***\n                    (0.2080) \n=============================\nStandard errors in\nparentheses.\n* p<.1, ** p<.05, ***p<.01'
    assert_equal(str(out), s)