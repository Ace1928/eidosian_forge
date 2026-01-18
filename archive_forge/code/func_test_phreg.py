import numpy as np
import pandas as pd
import pytest
from statsmodels.imputation import mice
import statsmodels.api as sm
from numpy.testing import assert_equal, assert_allclose
import warnings
def test_phreg(self):
    gen = np.random.RandomState(8742)
    n = 300
    x1 = gen.normal(size=n)
    x2 = gen.normal(size=n)
    event_time = gen.exponential(size=n) * np.exp(x1)
    obs_time = gen.exponential(size=n)
    time = np.where(event_time < obs_time, event_time, obs_time)
    status = np.where(time == event_time, 1, 0)
    df = pd.DataFrame({'time': time, 'status': status, 'x1': x1, 'x2': x2})
    df.loc[10:40, 'time'] = np.nan
    df.loc[10:40, 'status'] = np.nan
    df.loc[30:50, 'x1'] = np.nan
    df.loc[40:60, 'x2'] = np.nan
    from statsmodels.duration.hazard_regression import PHReg
    hist = []

    def cb(imp):
        hist.append(imp.data.shape)
    for pm in ('gaussian', 'boot'):
        idata = mice.MICEData(df, perturbation_method=pm, history_callback=cb)
        idata.set_imputer('time', '0 + x1 + x2', model_class=PHReg, init_kwds={'status': mice.PatsyFormula('status')}, predict_kwds={'pred_type': 'hr'}, perturbation_method=pm)
        x = idata.next_sample()
        assert isinstance(x, pd.DataFrame)
    assert all([val == (299, 4) for val in hist])