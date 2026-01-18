import os
import numpy as np
from statsmodels.duration.survfunc import (
from numpy.testing import assert_allclose
import pandas as pd
import pytest
def test_simultaneous_cb():
    df = bmt.loc[bmt['Group'] == 'ALL', :]
    sf = SurvfuncRight(df['T'], df['Status'])
    lcb1, ucb1 = sf.simultaneous_cb(transform='log')
    lcb2, ucb2 = sf.simultaneous_cb(transform='arcsin')
    ti = sf.surv_times.tolist()
    ix = [ti.index(x) for x in (110, 122, 129, 172)]
    assert_allclose(lcb1[ix], np.r_[0.43590582, 0.42115592, 0.4035897, 0.38785927])
    assert_allclose(ucb1[ix], np.r_[0.93491636, 0.89776803, 0.87922239, 0.85894181])
    assert_allclose(lcb2[ix], np.r_[0.52115708, 0.48079378, 0.45595321, 0.43341115])
    assert_allclose(ucb2[ix], np.r_[0.96465636, 0.92745068, 0.90885428, 0.88796708])