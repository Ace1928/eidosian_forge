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
def test_plot_ccf(close_figures):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ar = np.r_[1.0, -0.9]
    ma = np.r_[1.0, 0.9]
    armaprocess = tsp.ArmaProcess(ar, ma)
    rs = np.random.RandomState(1234)
    x1 = armaprocess.generate_sample(100, distrvs=rs.standard_normal)
    x2 = armaprocess.generate_sample(100, distrvs=rs.standard_normal)
    plot_ccf(x1, x2)
    plot_ccf(x1, x2, ax=ax, lags=10)
    plot_ccf(x1, x2, ax=ax)
    plot_ccf(x1, x2, ax=ax, alpha=None)
    plot_ccf(x1, x2, ax=ax, negative_lags=True)
    plot_ccf(x1, x2, ax=ax, adjusted=True)
    plot_ccf(x1, x2, ax=ax, fft=True)
    plot_ccf(x1, x2, ax=ax, title='CCF')
    plot_ccf(x1, x2, ax=ax, auto_ylims=True)
    plot_ccf(x1, x2, ax=ax, use_vlines=False)