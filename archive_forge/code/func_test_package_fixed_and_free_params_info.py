import numpy as np
import pytest
from numpy.testing import assert_allclose
from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import lake
from statsmodels.tsa.arima.estimators.hannan_rissanen import (
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tools.tools import Bunch
@pytest.mark.parametrize('fixed_params, spec_ar_lags, spec_ma_lags, expected_bunch', [({}, [1], [], Bunch(fixed_ar_lags=[], fixed_ma_lags=[], free_ar_lags=[1], free_ma_lags=[], fixed_ar_ix=np.array([], dtype=int), fixed_ma_ix=np.array([], dtype=int), free_ar_ix=np.array([0], dtype=int), free_ma_ix=np.array([], dtype=int), fixed_ar_params=np.array([]), fixed_ma_params=np.array([]))), ({'ar.L2': 0.1, 'ma.L1': 0.2}, [2], [1, 3], Bunch(fixed_ar_lags=[2], fixed_ma_lags=[1], free_ar_lags=[], free_ma_lags=[3], fixed_ar_ix=np.array([1], dtype=int), fixed_ma_ix=np.array([0], dtype=int), free_ar_ix=np.array([], dtype=int), free_ma_ix=np.array([2], dtype=int), fixed_ar_params=np.array([0.1]), fixed_ma_params=np.array([0.2]))), ({'ma.L5': 0.1, 'ma.L10': 0.2}, [], [5, 10], Bunch(fixed_ar_lags=[], fixed_ma_lags=[5, 10], free_ar_lags=[], free_ma_lags=[], fixed_ar_ix=np.array([], dtype=int), fixed_ma_ix=np.array([4, 9], dtype=int), free_ar_ix=np.array([], dtype=int), free_ma_ix=np.array([], dtype=int), fixed_ar_params=np.array([]), fixed_ma_params=np.array([0.1, 0.2])))])
def test_package_fixed_and_free_params_info(fixed_params, spec_ar_lags, spec_ma_lags, expected_bunch):
    actual_bunch = _package_fixed_and_free_params_info(fixed_params, spec_ar_lags, spec_ma_lags)
    assert isinstance(actual_bunch, Bunch)
    assert len(actual_bunch) == len(expected_bunch)
    assert actual_bunch.keys() == expected_bunch.keys()
    lags = ['fixed_ar_lags', 'fixed_ma_lags', 'free_ar_lags', 'free_ma_lags']
    for k in lags:
        assert isinstance(actual_bunch[k], list)
        assert actual_bunch[k] == expected_bunch[k]
    ixs = ['fixed_ar_ix', 'fixed_ma_ix', 'free_ar_ix', 'free_ma_ix']
    for k in ixs:
        assert isinstance(actual_bunch[k], np.ndarray)
        assert actual_bunch[k].dtype in [np.int64, np.int32]
        np.testing.assert_array_equal(actual_bunch[k], expected_bunch[k])
    params = ['fixed_ar_params', 'fixed_ma_params']
    for k in params:
        assert isinstance(actual_bunch[k], np.ndarray)
        np.testing.assert_array_equal(actual_bunch[k], expected_bunch[k])