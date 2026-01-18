import os
import numpy as np
from statsmodels.duration.survfunc import (
from numpy.testing import assert_allclose
import pandas as pd
import pytest
def test_survfunc_entry_2():
    times = np.r_[1, 3, 3, 5, 5, 7, 7, 8, 8, 9, 10, 10]
    status = np.r_[1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1]
    entry = np.r_[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    sf = SurvfuncRight(times, status, entry=entry)
    sf0 = SurvfuncRight(times, status)
    assert_allclose(sf.n_risk, sf0.n_risk)
    assert_allclose(sf.surv_times, sf0.surv_times)
    assert_allclose(sf.surv_prob, sf0.surv_prob)
    assert_allclose(sf.surv_prob_se, sf0.surv_prob_se)