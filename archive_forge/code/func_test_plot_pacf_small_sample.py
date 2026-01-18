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
def test_plot_pacf_small_sample():
    idx = [pd.Timestamp.now() + pd.Timedelta(seconds=i) for i in range(10)]
    df = pd.DataFrame(index=idx, columns=['a'], data=list(range(10)))
    plot_pacf(df)