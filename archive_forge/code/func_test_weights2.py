import os
import numpy as np
from statsmodels.duration.survfunc import (
from numpy.testing import assert_allclose
import pandas as pd
import pytest
def test_weights2():
    tm = np.r_[1, 3, 5, 6, 7, 2, 4, 6, 8, 10]
    st = np.r_[1, 1, 0, 1, 1, 1, 1, 0, 1, 1]
    wt = np.r_[1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    tm0 = np.r_[1, 3, 5, 6, 7, 2, 4, 6, 8, 10, 2, 4, 6, 8, 10]
    st0 = np.r_[1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1]
    sf0 = SurvfuncRight(tm, st, freq_weights=wt)
    sf1 = SurvfuncRight(tm0, st0)
    assert_allclose(sf0.surv_times, sf1.surv_times)
    assert_allclose(sf0.surv_prob, sf1.surv_prob)
    assert_allclose(sf0.surv_prob_se, np.r_[0.06666667, 0.1210311, 0.14694547, 0.19524829, 0.23183377, 0.30618115, 0.46770386, 0.84778942])