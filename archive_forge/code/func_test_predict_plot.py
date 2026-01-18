from statsmodels.compat.pandas import MONTH_END
from statsmodels.compat.python import lmap
import calendar
from io import BytesIO
import locale
import numpy as np
from numpy.testing import assert_, assert_equal
import pandas as pd
import pytest
from statsmodels.datasets import elnino, macrodata
from statsmodels.graphics.tsaplots import (
from statsmodels.tsa import arima_process as tsp
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
@pytest.mark.matplotlib
@pytest.mark.parametrize('model_and_args', [(AutoReg, dict(lags=2, old_names=False)), (ARIMA, dict(order=(2, 0, 0)))])
@pytest.mark.parametrize('use_pandas', [True, False])
@pytest.mark.parametrize('alpha', [None, 0.1])
def test_predict_plot(use_pandas, model_and_args, alpha):
    model, kwargs = model_and_args
    rs = np.random.RandomState(0)
    y = rs.standard_normal(1000)
    for i in range(2, 1000):
        y[i] += 1.8 * y[i - 1] - 0.9 * y[i - 2]
    y = y[100:]
    if use_pandas:
        index = pd.date_range('1960-1-1', freq=MONTH_END, periods=y.shape[0] + 24)
        start = index[index.shape[0] // 2]
        end = index[-1]
        y = pd.Series(y, index=index[:-24])
    else:
        start = y.shape[0] // 2
        end = y.shape[0] + 24
    res = model(y, **kwargs).fit()
    fig = plot_predict(res, start, end, alpha=alpha)
    assert isinstance(fig, plt.Figure)