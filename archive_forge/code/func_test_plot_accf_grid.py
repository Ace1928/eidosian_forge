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
def test_plot_accf_grid(close_figures):
    fig = plt.figure()
    ar = np.r_[1.0, -0.9]
    ma = np.r_[1.0, 0.9]
    armaprocess = tsp.ArmaProcess(ar, ma)
    rs = np.random.RandomState(1234)
    x = np.vstack([armaprocess.generate_sample(100, distrvs=rs.standard_normal), armaprocess.generate_sample(100, distrvs=rs.standard_normal)]).T
    plot_accf_grid(x)
    plot_accf_grid(pd.DataFrame({'x': x[:, 0], 'y': x[:, 1]}))
    plot_accf_grid(x, fig=fig, lags=10)
    plot_accf_grid(x, fig=fig)
    plot_accf_grid(x, fig=fig, negative_lags=False)
    plot_accf_grid(x, fig=fig, alpha=None)
    plot_accf_grid(x, fig=fig, adjusted=True)
    plot_accf_grid(x, fig=fig, fft=True)
    plot_accf_grid(x, fig=fig, auto_ylims=True)
    plot_accf_grid(x, fig=fig, use_vlines=False)