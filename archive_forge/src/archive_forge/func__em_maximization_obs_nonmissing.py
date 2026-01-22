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
def _em_maximization_obs_nonmissing(self, res, Eaa, a, compute_H=False):
    """EM maximization step, observation equation without missing data."""
    s = self._s
    dtype = Eaa.dtype
    k = s.k_states_factors
    Lambda = np.zeros((self.k_endog, k), dtype=dtype)
    for i in range(self.k_endog):
        y = self.endog[:, i:i + 1]
        iloc = self._s.endog_factor_iloc[i]
        factor_ix = s['factors_L1'][iloc]
        ix = (np.s_[:],) + np.ix_(factor_ix, factor_ix)
        A = Eaa[ix].sum(axis=0)
        B = y.T @ a[:, factor_ix, 0]
        if self.idiosyncratic_ar1:
            ix1 = s.k_states_factors + i
            ix2 = ix1 + 1
            B -= Eaa[:, ix1:ix2, factor_ix].sum(axis=0)
        try:
            Lambda[i, factor_ix] = cho_solve(cho_factor(A), B.T).T
        except LinAlgError:
            Lambda[i, factor_ix] = np.linalg.solve(A, B.T).T
    if compute_H:
        Z = self['design'].copy()
        Z[:, :k] = Lambda
        BL = self.endog.T @ a[..., 0] @ Z.T
        C = self.endog.T @ self.endog
        H = (C + -BL - BL.T + Z @ Eaa.sum(axis=0) @ Z.T) / self.nobs
    else:
        H = np.zeros((self.k_endog, self.k_endog), dtype=dtype) * np.nan
    return (Lambda, H)