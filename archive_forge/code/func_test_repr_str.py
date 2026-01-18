import numpy as np
import pandas as pd
from numpy.testing import assert_, assert_equal, assert_allclose, assert_raises
from statsmodels.tsa.arima import specification, params
def test_repr_str():
    exog = pd.DataFrame([[0, 0]], columns=['a', 'b'])
    spec = specification.SARIMAXSpecification(exog=exog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
    p = params.SARIMAXParams(spec=spec)
    assert_equal(repr(p), 'SARIMAXParams(exog=[nan nan], ar=[nan], ma=[nan], seasonal_ar=[nan], seasonal_ma=[nan], sigma2=nan)')
    p.exog_params = [1, 2]
    assert_equal(repr(p), 'SARIMAXParams(exog=[1. 2.], ar=[nan], ma=[nan], seasonal_ar=[nan], seasonal_ma=[nan], sigma2=nan)')
    p.ar_params = [0.5]
    assert_equal(repr(p), 'SARIMAXParams(exog=[1. 2.], ar=[0.5], ma=[nan], seasonal_ar=[nan], seasonal_ma=[nan], sigma2=nan)')
    p.ma_params = [0.2]
    assert_equal(repr(p), 'SARIMAXParams(exog=[1. 2.], ar=[0.5], ma=[0.2], seasonal_ar=[nan], seasonal_ma=[nan], sigma2=nan)')
    p.seasonal_ar_params = [0.001]
    assert_equal(repr(p), 'SARIMAXParams(exog=[1. 2.], ar=[0.5], ma=[0.2], seasonal_ar=[0.001], seasonal_ma=[nan], sigma2=nan)')
    p.seasonal_ma_params = [-0.001]
    assert_equal(repr(p), 'SARIMAXParams(exog=[1. 2.], ar=[0.5], ma=[0.2], seasonal_ar=[0.001], seasonal_ma=[-0.001], sigma2=nan)')
    p.sigma2 = 10.123
    assert_equal(repr(p), 'SARIMAXParams(exog=[1. 2.], ar=[0.5], ma=[0.2], seasonal_ar=[0.001], seasonal_ma=[-0.001], sigma2=10.123)')