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
def test_plot_month(close_figures):
    dta = elnino.load_pandas().data
    dta['YEAR'] = dta.YEAR.astype(int).apply(str)
    dta = dta.set_index('YEAR').T.unstack()
    dates = pd.to_datetime(['-'.join([x[1], x[0]]) for x in dta.index.values], format='%b-%Y')
    fig = month_plot(dta.values, dates=dates, ylabel='el nino')
    dta.index = pd.DatetimeIndex(dates)
    fig = month_plot(dta)
    dta.index = pd.DatetimeIndex(dates, freq='MS')
    fig = month_plot(dta)
    dta.index = pd.PeriodIndex(dates, freq='M')
    fig = month_plot(dta)
    try:
        with calendar.different_locale('DE_de'):
            fig = month_plot(dta)
            labels = [_.get_text() for _ in fig.axes[0].get_xticklabels()]
            expected = ['Jan', 'Feb', ('Mär', 'Mrz'), 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez']
            for lbl, exp in zip(labels, expected):
                if isinstance(exp, tuple):
                    assert lbl in exp
                else:
                    assert lbl == exp
    except locale.Error:
        pytest.xfail(reason='Failure due to unsupported locale')