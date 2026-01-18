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
def test_plot_quarter(close_figures):
    dta = macrodata.load_pandas().data
    dates = lmap('-Q'.join, zip(dta.year.astype(int).apply(str), dta.quarter.astype(int).apply(str)))
    quarter_plot(dta.unemp.values, dates)
    dta.set_index(pd.DatetimeIndex(dates, freq='QS-OCT'), inplace=True)
    quarter_plot(dta.unemp)
    dta.index = pd.DatetimeIndex(dates, freq='QS-OCT')
    quarter_plot(dta.unemp)
    dta.index = pd.PeriodIndex(dates, freq='Q')
    quarter_plot(dta.unemp)