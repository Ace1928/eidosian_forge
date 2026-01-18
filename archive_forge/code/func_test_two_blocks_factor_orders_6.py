from statsmodels.compat.pandas import QUARTER_END
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.signal import lfilter
from statsmodels.tsa.statespace import (
@pytest.mark.skip(reason='Monte carlo test, very slow, kept for manual runs')
def test_two_blocks_factor_orders_6(reset_randomstate):
    nobs = 1000
    idiosyncratic_ar1 = True
    k1 = 3
    k2 = 10
    endog1_M, endog1_Q, f1 = gen_k_factor1(nobs, k=k1, idiosyncratic_ar1=idiosyncratic_ar1)
    endog2_M, endog2_Q, f2 = gen_k_factor2(nobs, k=k2, idiosyncratic_ar1=idiosyncratic_ar1)
    endog_M = pd.concat([endog1_M, f2, endog2_M], axis=1)
    endog_Q = pd.concat([endog1_Q, endog2_Q], axis=1)
    factors = {f'yM{i + 1}_f1': ['a'] for i in range(k1)}
    factors.update({f'yQ{i + 1}_f1': ['a'] for i in range(k1)})
    factors.update({f'f{i + 1}': ['b'] for i in range(2)})
    factors.update({f'yM{i + 1}_f2': ['b'] for i in range(k2)})
    factors.update({f'yQ{i + 1}_f2': ['b'] for i in range(k2)})
    factor_multiplicities = {'b': 2}
    mod = dynamic_factor_mq.DynamicFactorMQ(endog_M, endog_quarterly=endog_Q, factors=factors, factor_multiplicities=factor_multiplicities, factor_orders=6, idiosyncratic_ar1=idiosyncratic_ar1, standardize=False)
    mod.fit()
    from scipy.linalg import block_diag
    M1 = np.kron(np.eye(6), mod['design', 3:5, :2])
    M2 = np.kron(np.eye(6), mod['design', 0:1, 12:13])
    M = block_diag(M1, M2)
    Mi = np.linalg.inv(M)
    Z = mod['design', :, :18]
    A = mod['transition', :18, :18]
    R = mod['selection', :18, :3]
    Q = block_diag(mod['state_cov', :2, :2], mod['state_cov', 12:13, 12:13])
    RQR = R @ Q @ R.T
    Z2 = Z @ Mi
    A2 = M @ A @ Mi
    Q2 = M @ RQR @ M.T
    print(Z2.round(2))
    print(A2.round(2))
    print(Q2.round(2))