import numpy as np
from scipy.signal import lfilter
from statsmodels.tools.tools import Bunch
from statsmodels.regression.linear_model import OLS, yule_walker
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tsa.arima.params import SARIMAXParams

    Stitch together fixed and free params, by the order of lags, for setting
    SARIMAXParams.ma_params or SARIMAXParams.ar_params

    Parameters
    ----------
    fixed_ar_or_ma_lags : list or np.array
    fixed_ar_or_ma_params : list or np.array
        fixed_ar_or_ma_params corresponds with fixed_ar_or_ma_lags
    free_ar_or_ma_lags : list or np.array
    free_ar_or_ma_params : list or np.array
        free_ar_or_ma_params corresponds with free_ar_or_ma_lags
    spec_ar_or_ma_lags : list
        SARIMAXSpecification.ar_lags or SARIMAXSpecification.ma_lags

    Returns
    -------
    list of fixed and free params by the order of lags
    