import numpy as np
from scipy.signal import lfilter
from statsmodels.tools.tools import Bunch
from statsmodels.regression.linear_model import OLS, yule_walker
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tsa.arima.params import SARIMAXParams
def _package_fixed_and_free_params_info(fixed_params, spec_ar_lags, spec_ma_lags):
    """
    Parameters
    ----------
    fixed_params : dict
    spec_ar_lags : list of int
        SARIMAXSpecification.ar_lags
    spec_ma_lags : list of int
        SARIMAXSpecification.ma_lags

    Returns
    -------
    Bunch with
    (lags) fixed_ar_lags, fixed_ma_lags, free_ar_lags, free_ma_lags;
    (ix) fixed_ar_ix, fixed_ma_ix, free_ar_ix, free_ma_ix;
    (params) fixed_ar_params, free_ma_params
    """
    fixed_ar_lags_and_params = []
    fixed_ma_lags_and_params = []
    for key, val in fixed_params.items():
        lag = int(key.split('.')[-1].lstrip('L'))
        if key.startswith('ar'):
            fixed_ar_lags_and_params.append((lag, val))
        elif key.startswith('ma'):
            fixed_ma_lags_and_params.append((lag, val))
    fixed_ar_lags_and_params.sort()
    fixed_ma_lags_and_params.sort()
    fixed_ar_lags = [lag for lag, _ in fixed_ar_lags_and_params]
    fixed_ar_params = np.array([val for _, val in fixed_ar_lags_and_params])
    fixed_ma_lags = [lag for lag, _ in fixed_ma_lags_and_params]
    fixed_ma_params = np.array([val for _, val in fixed_ma_lags_and_params])
    free_ar_lags = [lag for lag in spec_ar_lags if lag not in set(fixed_ar_lags)]
    free_ma_lags = [lag for lag in spec_ma_lags if lag not in set(fixed_ma_lags)]
    free_ar_ix = np.array(free_ar_lags, dtype=int) - 1
    free_ma_ix = np.array(free_ma_lags, dtype=int) - 1
    fixed_ar_ix = np.array(fixed_ar_lags, dtype=int) - 1
    fixed_ma_ix = np.array(fixed_ma_lags, dtype=int) - 1
    return Bunch(fixed_ar_lags=fixed_ar_lags, fixed_ma_lags=fixed_ma_lags, free_ar_lags=free_ar_lags, free_ma_lags=free_ma_lags, fixed_ar_ix=fixed_ar_ix, fixed_ma_ix=fixed_ma_ix, free_ar_ix=free_ar_ix, free_ma_ix=free_ma_ix, fixed_ar_params=fixed_ar_params, fixed_ma_params=fixed_ma_params)