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
def test_plot_acf_missing(close_figures):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ar = np.r_[1.0, -0.9]
    ma = np.r_[1.0, 0.9]
    armaprocess = tsp.ArmaProcess(ar, ma)
    rs = np.random.RandomState(1234)
    acf = armaprocess.generate_sample(100, distrvs=rs.standard_normal)
    acf[::13] = np.nan
    buff = BytesIO()
    plot_acf(acf, ax=ax, missing='drop')
    fig.savefig(buff, format='rgba')
    buff.seek(0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    buff_conservative = BytesIO()
    plot_acf(acf, ax=ax, missing='conservative')
    fig.savefig(buff_conservative, format='rgba')
    buff_conservative.seek(0)
    assert_(buff.read() != buff_conservative.read())