import numpy as np
import pytest
from numpy.testing import assert_allclose
from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import lake
from statsmodels.tsa.arima.estimators.hannan_rissanen import (
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tools.tools import Bunch
@pytest.mark.parametrize('ar_order, ma_order, fixed_params, invalid_fixed_params', [(2, [1, 0, 1], None, None), ([0, 1], 0, {}, None), (1, 3, {'ar.L2': 1, 'ma.L2': 0}, ['ar.L2']), ([0, 1], [0, 0, 1], {'ma.L1': 0, 'sigma2': 1}, ['ma.L2', 'sigma2']), (0, 0, {'ma.L1': 0, 'ar.L1': 0}, ['ar.L1', 'ma.L1']), (5, [1, 0], {'random_param': 0, 'ar.L1': 0}, ['random_param']), (0, 2, {'ma.L1': -1, 'ma.L2': 1}, None), (1, 0, {'ar.L1': 0}, None), ([1, 0, 1], 3, {'ma.L2': 1, 'ar.L3': -1}, None), (2, 2, {'ma.L1': 1, 'ma.L2': 1, 'ar.L1': 1, 'ar.L2': 1}, None)])
def test_validate_fixed_params(ar_order, ma_order, fixed_params, invalid_fixed_params):
    endog = np.random.normal(size=100)
    spec = SARIMAXSpecification(endog, ar_order=ar_order, ma_order=ma_order)
    if invalid_fixed_params is None:
        _validate_fixed_params(fixed_params, spec.param_names)
        hannan_rissanen(endog, ar_order=ar_order, ma_order=ma_order, fixed_params=fixed_params, unbiased=False)
    else:
        valid_params = sorted(list(set(spec.param_names) - {'sigma2'}))
        msg = f'Invalid fixed parameter(s): {invalid_fixed_params}. Please select among {valid_params}.'
        with pytest.raises(ValueError) as e:
            _validate_fixed_params(fixed_params, spec.param_names)
            assert e.msg == msg
        with pytest.raises(ValueError) as e:
            hannan_rissanen(endog, ar_order=ar_order, ma_order=ma_order, fixed_params=fixed_params, unbiased=False)
            assert e.msg == msg