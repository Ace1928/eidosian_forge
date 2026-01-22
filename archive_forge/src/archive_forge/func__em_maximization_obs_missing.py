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
def _em_maximization_obs_missing(self, res, Eaa, a, compute_H=False):
    """EM maximization step, observation equation with missing data."""
    s = self._s
    dtype = Eaa.dtype
    k = s.k_states_factors
    Lambda = np.zeros((self.k_endog, k), dtype=dtype)
    W = 1 - res.missing.T
    mask = W.astype(bool)
    for i in range(self.k_endog_M):
        iloc = self._s.endog_factor_iloc[i]
        factor_ix = s['factors_L1'][iloc]
        m = mask[:, i]
        yt = self.endog[m, i:i + 1]
        ix = np.ix_(m, factor_ix, factor_ix)
        Ai = Eaa[ix].sum(axis=0)
        Bi = yt.T @ a[np.ix_(m, factor_ix)][..., 0]
        if self.idiosyncratic_ar1:
            ix1 = s.k_states_factors + i
            ix2 = ix1 + 1
            Bi -= Eaa[m, ix1:ix2][..., factor_ix].sum(axis=0)
        try:
            Lambda[i, factor_ix] = cho_solve(cho_factor(Ai), Bi.T).T
        except LinAlgError:
            Lambda[i, factor_ix] = np.linalg.solve(Ai, Bi.T).T
    if self.k_endog_Q > 0:
        multipliers = np.array([1, 2, 3, 2, 1])[:, None]
        for i in range(self.k_endog_M, self.k_endog):
            iloc = self._s.endog_factor_iloc[i]
            factor_ix = s['factors_L1_5_ix'][:, iloc].ravel().tolist()
            R, _ = self.loading_constraints(i)
            iQ = i - self.k_endog_M
            m = mask[:, i]
            yt = self.endog[m, i:i + 1]
            ix = np.ix_(m, factor_ix, factor_ix)
            Ai = Eaa[ix].sum(axis=0)
            BiQ = yt.T @ a[np.ix_(m, factor_ix)][..., 0]
            if self.idiosyncratic_ar1:
                ix = (np.s_[:],) + np.ix_(s['idio_ar_Q_ix'][iQ], factor_ix)
                Eepsf = Eaa[ix]
                BiQ -= (multipliers * Eepsf[m].sum(axis=0)).sum(axis=0)
            try:
                L_and_lower = cho_factor(Ai)
                unrestricted = cho_solve(L_and_lower, BiQ.T).T[0]
                AiiRT = cho_solve(L_and_lower, R.T)
                L_and_lower = cho_factor(R @ AiiRT)
                RAiiRTiR = cho_solve(L_and_lower, R)
                restricted = unrestricted - AiiRT @ RAiiRTiR @ unrestricted
            except LinAlgError:
                Aii = np.linalg.inv(Ai)
                unrestricted = (BiQ @ Aii)[0]
                RARi = np.linalg.inv(R @ Aii @ R.T)
                restricted = unrestricted - Aii @ R.T @ RARi @ R @ unrestricted
            Lambda[i, factor_ix] = restricted
    if compute_H:
        Z = self['design'].copy()
        Z[:, :Lambda.shape[1]] = Lambda
        y = np.nan_to_num(self.endog)
        C = y.T @ y
        W = W[..., None]
        IW = 1 - W
        WL = W * Z
        WLT = WL.transpose(0, 2, 1)
        BL = y[..., None] @ a.transpose(0, 2, 1) @ WLT
        A = Eaa
        BLT = BL.transpose(0, 2, 1)
        IWT = IW.transpose(0, 2, 1)
        H = (C + (-BL - BLT + WL @ A @ WLT + IW * self['obs_cov'] * IWT).sum(axis=0)) / self.nobs
    else:
        H = np.zeros((self.k_endog, self.k_endog), dtype=dtype) * np.nan
    return (Lambda, H)