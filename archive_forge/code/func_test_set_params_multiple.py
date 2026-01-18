import numpy as np
import pandas as pd
from numpy.testing import assert_, assert_equal, assert_allclose, assert_raises
from statsmodels.tsa.arima import specification, params
def test_set_params_multiple():
    exog = pd.DataFrame([[0, 0]], columns=['a', 'b'])
    spec = specification.SARIMAXSpecification(exog=exog, order=(2, 1, 2), seasonal_order=(2, 1, 2, 4))
    p = params.SARIMAXParams(spec=spec)
    p.params = [-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11]
    assert_equal(p.params, [-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11])
    assert_equal(p.exog_params, [-1, 2])
    assert_equal(p.ar_params, [-3, 4])
    assert_equal(p.ma_params, [-5, 6])
    assert_equal(p.seasonal_ar_params, [-7, 8])
    assert_equal(p.seasonal_ma_params, [-9, 10])
    assert_equal(p.sigma2, -11)
    assert_equal(p.ar_poly.coef, np.r_[1, 3, -4])
    assert_equal(p.ma_poly.coef, np.r_[1, -5, 6])
    assert_equal(p.seasonal_ar_poly.coef, np.r_[1, 0, 0, 0, 7, 0, 0, 0, -8])
    assert_equal(p.seasonal_ma_poly.coef, np.r_[1, 0, 0, 0, -9, 0, 0, 0, 10])
    assert_equal(p.reduced_ar_poly.coef, [1, 3, -4, 0, 7, -3 * -7, 4 * -7, 0, -8, -3 * 8, 4 * 8])
    assert_equal(p.reduced_ma_poly.coef, [1, -5, 6, 0, -9, -5 * -9, 6 * -9, 0, 10, -5 * 10, 6 * 10])