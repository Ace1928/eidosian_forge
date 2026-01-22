from statsmodels.compat.pandas import MONTH_END, QUARTER_END
from collections import OrderedDict
from warnings import warn
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.validation import int_like
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.multivariate.pca import PCA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace._quarterly_ar1 import QuarterlyAR1
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tools.tools import Bunch
from statsmodels.tools.validation import string_like
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.statespace import mlemodel, initialization
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.statespace.kalman_smoother import (
from statsmodels.base.data import PandasData
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.tableformatting import fmt_params
def _em_maximization_step(self, res, params0, mstep_method=None):
    """EM maximization step."""
    s = self._s
    a = res.smoothed_state.T[..., None]
    cov_a = res.smoothed_state_cov.transpose(2, 0, 1)
    acov_a = res.smoothed_state_autocov.transpose(2, 0, 1)
    Eaa = cov_a.copy() + np.matmul(a, a.transpose(0, 2, 1))
    Eaa1 = acov_a[:-1] + np.matmul(a[1:], a[:-1].transpose(0, 2, 1))
    has_missing = np.any(res.nmissing)
    if mstep_method is None:
        mstep_method = 'missing' if has_missing else 'nonmissing'
    mstep_method = mstep_method.lower()
    if mstep_method == 'nonmissing' and has_missing:
        raise ValueError('Cannot use EM algorithm option `mstep_method="nonmissing"` with missing data.')
    if mstep_method == 'nonmissing':
        func = self._em_maximization_obs_nonmissing
    elif mstep_method == 'missing':
        func = self._em_maximization_obs_missing
    else:
        raise ValueError('Invalid maximization step method: "%s".' % mstep_method)
    Lambda, H = func(res, Eaa, a, compute_H=not self.idiosyncratic_ar1)
    factor_ar = []
    factor_cov = []
    for b in s.factor_blocks:
        A = Eaa[:-1, b['factors_ar'], b['factors_ar']].sum(axis=0)
        B = Eaa1[:, b['factors_L1'], b['factors_ar']].sum(axis=0)
        C = Eaa[1:, b['factors_L1'], b['factors_L1']].sum(axis=0)
        nobs = Eaa.shape[0] - 1
        try:
            f_A = cho_solve(cho_factor(A), B.T).T
        except LinAlgError:
            f_A = np.linalg.solve(A, B.T).T
        f_Q = (C - f_A @ B.T) / nobs
        factor_ar += f_A.ravel().tolist()
        factor_cov += np.linalg.cholesky(f_Q)[np.tril_indices_from(f_Q)].tolist()
    if self.idiosyncratic_ar1:
        ix = s['idio_ar_L1']
        Ad = Eaa[:-1, ix, ix].sum(axis=0).diagonal()
        Bd = Eaa1[:, ix, ix].sum(axis=0).diagonal()
        Cd = Eaa[1:, ix, ix].sum(axis=0).diagonal()
        nobs = Eaa.shape[0] - 1
        alpha = Bd / Ad
        sigma2 = (Cd - alpha * Bd) / nobs
    else:
        ix = s['idio_ar_L1']
        C = Eaa[:, ix, ix].sum(axis=0)
        sigma2 = np.r_[H.diagonal()[self._o['M']], C.diagonal() / Eaa.shape[0]]
    params1 = np.zeros_like(params0)
    loadings = []
    for i in range(self.k_endog):
        iloc = self._s.endog_factor_iloc[i]
        factor_ix = s['factors_L1'][iloc]
        loadings += Lambda[i, factor_ix].tolist()
    params1[self._p['loadings']] = loadings
    params1[self._p['factor_ar']] = factor_ar
    params1[self._p['factor_cov']] = factor_cov
    if self.idiosyncratic_ar1:
        params1[self._p['idiosyncratic_ar1']] = alpha
    params1[self._p['idiosyncratic_var']] = sigma2
    return params1