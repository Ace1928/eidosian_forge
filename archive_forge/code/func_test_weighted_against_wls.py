from io import BytesIO
from itertools import product
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from statsmodels import tools
from statsmodels.regression.linear_model import WLS
from statsmodels.regression.rolling import RollingWLS, RollingOLS
def test_weighted_against_wls(weighted_data):
    y, x, w = weighted_data
    mod = RollingWLS(y, x, weights=w, window=100)
    res = mod.fit(use_t=True)
    for i in range(100, y.shape[0]):
        _y = get_sub(y, i, 100)
        _x = get_sub(x, i, 100)
        if w is not None:
            _w = get_sub(w, i, 100)
        else:
            _w = np.ones_like(_y)
        wls = WLS(_y, _x, weights=_w, missing='drop').fit()
        rolling_params = get_single(res.params, i - 1)
        rolling_nobs = get_single(res.nobs, i - 1)
        assert_allclose(rolling_params, wls.params)
        assert_allclose(rolling_nobs, wls.nobs)
        assert_allclose(get_single(res.ssr, i - 1), wls.ssr)
        assert_allclose(get_single(res.llf, i - 1), wls.llf)
        assert_allclose(get_single(res.aic, i - 1), wls.aic)
        assert_allclose(get_single(res.bic, i - 1), wls.bic)
        assert_allclose(get_single(res.centered_tss, i - 1), wls.centered_tss)
        assert_allclose(res.df_model, wls.df_model)
        assert_allclose(get_single(res.df_resid, i - 1), wls.df_resid)
        assert_allclose(get_single(res.ess, i - 1), wls.ess, atol=1e-08)
        assert_allclose(res.k_constant, wls.k_constant)
        assert_allclose(get_single(res.mse_model, i - 1), wls.mse_model)
        assert_allclose(get_single(res.mse_resid, i - 1), wls.mse_resid)
        assert_allclose(get_single(res.mse_total, i - 1), wls.mse_total)
        assert_allclose(get_single(res.rsquared, i - 1), wls.rsquared, atol=1e-08)
        assert_allclose(get_single(res.rsquared_adj, i - 1), wls.rsquared_adj, atol=1e-08)
        assert_allclose(get_single(res.uncentered_tss, i - 1), wls.uncentered_tss)