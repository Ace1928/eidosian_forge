import numpy as np
import pandas as pd
def _initialization_simple(endog, trend=False, seasonal=False, seasonal_periods=None):
    nobs = len(endog)
    initial_trend = None
    initial_seasonal = None
    if seasonal is None or not seasonal:
        initial_level = endog[0]
        if trend == 'add':
            initial_trend = endog[1] - endog[0]
        elif trend == 'mul':
            initial_trend = endog[1] / endog[0]
    else:
        if nobs < 2 * seasonal_periods:
            raise ValueError('Cannot compute initial seasonals using heuristic method with less than two full seasonal cycles in the data.')
        initial_level = np.mean(endog[:seasonal_periods])
        m = seasonal_periods
        if trend is not None:
            initial_trend = (pd.Series(endog).diff(m)[m:2 * m] / m).mean()
        if seasonal == 'add':
            initial_seasonal = endog[:m] - initial_level
        elif seasonal == 'mul':
            initial_seasonal = endog[:m] / initial_level
    return (initial_level, initial_trend, initial_seasonal)