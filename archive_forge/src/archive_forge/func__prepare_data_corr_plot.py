from statsmodels.compat.pandas import deprecate_kwarg
import calendar
import numpy as np
import pandas as pd
from statsmodels.graphics import utils
from statsmodels.tools.validation import array_like
from statsmodels.tsa.stattools import acf, pacf, ccf
def _prepare_data_corr_plot(x, lags, zero):
    zero = bool(zero)
    irregular = False if zero else True
    if lags is None:
        nobs = x.shape[0]
        lim = min(int(np.ceil(10 * np.log10(nobs))), nobs // 2)
        lags = np.arange(not zero, lim + 1)
    elif np.isscalar(lags):
        lags = np.arange(not zero, int(lags) + 1)
    else:
        irregular = True
        lags = np.asanyarray(lags).astype(int)
    nlags = lags.max(0)
    return (lags, nlags, irregular)