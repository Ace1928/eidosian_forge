import numpy as np
import pandas as pd
from numpy.testing import assert_, assert_equal, assert_allclose, assert_raises
from statsmodels.tsa.arima import specification, params
def test_set_poly_longer_lags():
    exog = pd.DataFrame([[0, 0]], columns=['a', 'b'])
    spec = specification.SARIMAXSpecification(exog=exog, order=(2, 1, 2), seasonal_order=(2, 1, 2, 4))
    p = params.SARIMAXParams(spec=spec)
    p.exog_params = [-1, 2]
    p.sigma2 = -11
    p.ar_poly = np.r_[1, 3, -4]
    p.ma_poly = np.r_[1, -5, 6]
    p.seasonal_ar_poly = np.r_[1, 0, 0, 0, 7, 0, 0, 0, -8]
    p.seasonal_ma_poly = np.r_[1, 0, 0, 0, -9, 0, 0, 0, 10]
    assert_equal(p.params, [-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11])
    assert_equal(p.exog_params, [-1, 2])
    assert_equal(p.ar_params, [-3, 4])
    assert_equal(p.ma_params, [-5, 6])
    assert_equal(p.seasonal_ar_params, [-7, 8])
    assert_equal(p.seasonal_ma_params, [-9, 10])
    assert_equal(p.sigma2, -11)