from statsmodels.compat.pandas import MONTH_END
from statsmodels.compat.pytest import pytest_warns
import datetime as dt
from itertools import product
from typing import NamedTuple, Union
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
from pandas import Index, Series, date_range, period_range
from pandas.testing import assert_series_equal
import pytest
from statsmodels.datasets import macrodata, sunspots
from statsmodels.iolib.summary import Summary
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.sm_exceptions import SpecificationWarning, ValueWarning
from statsmodels.tools.tools import Bunch
from statsmodels.tsa.ar_model import (
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.deterministic import (
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.tests.results import results_ar
def test_predict_exog():
    rs = np.random.RandomState(12345678)
    e = rs.standard_normal(1001)
    y = np.empty(1001)
    x = rs.standard_normal((1001, 2))
    y[:3] = e[:3] * np.sqrt(1.0 / (1 - 0.9 ** 2)) + x[:3].sum(1)
    for i in range(3, 1001):
        y[i] = 10 + 0.9 * y[i - 1] - 0.5 * y[i - 3] + e[i] + x[i].sum()
    ys = pd.Series(y, index=pd.date_range(dt.datetime(1950, 1, 1), periods=1001, freq=MONTH_END))
    xdf = pd.DataFrame(x, columns=['x0', 'x1'], index=ys.index)
    mod = AutoReg(ys, [1, 3], trend='c', exog=xdf)
    res = mod.fit()
    assert '-X' in str(res.summary())
    pred = res.predict(900)
    c = res.params.iloc[0]
    ar = res.params.iloc[1:3]
    ex = np.asarray(res.params.iloc[3:])
    phi_1 = ar.iloc[0]
    phi_2 = ar.iloc[1]
    direct = c + phi_1 * y[899:-1] + phi_2 * y[897:-3]
    direct += ex[0] * x[900:, 0] + ex[1] * x[900:, 1]
    idx = pd.date_range(ys.index[900], periods=101, freq=MONTH_END)
    direct = pd.Series(direct, index=idx)
    assert_series_equal(pred, direct)
    exog_oos = rs.standard_normal((100, 2))
    pred = res.predict(900, 1100, dynamic=True, exog_oos=exog_oos)
    direct = np.zeros(201)
    phi_1 = ar.iloc[0]
    phi_2 = ar.iloc[1]
    direct[0] = c + phi_1 * y[899] + phi_2 * y[897] + x[900] @ ex
    direct[1] = c + phi_1 * direct[0] + phi_2 * y[898] + x[901] @ ex
    direct[2] = c + phi_1 * direct[1] + phi_2 * y[899] + x[902] @ ex
    for i in range(3, 201):
        direct[i] = c + phi_1 * direct[i - 1] + phi_2 * direct[i - 3]
        if 900 + i < x.shape[0]:
            direct[i] += x[900 + i] @ ex
        else:
            direct[i] += exog_oos[i - 101] @ ex
    direct = pd.Series(direct, index=pd.date_range(ys.index[900], periods=201, freq=MONTH_END))
    assert_series_equal(pred, direct)