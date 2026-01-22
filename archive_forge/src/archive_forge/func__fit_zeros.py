from __future__ import annotations
from statsmodels.compat.python import lzip
from functools import reduce
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.base.data import handle_data
from statsmodels.base.optimizer import Optimizer
import statsmodels.base.wrapper as wrap
from statsmodels.formula import handle_formula_data
from statsmodels.stats.contrast import (
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.decorators import (
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import nan_dot, recipr
from statsmodels.tools.validation import bool_like
def _fit_zeros(self, keep_index=None, start_params=None, return_auxiliary=False, k_params=None, **fit_kwds):
    """experimental, fit the model subject to zero constraints

        Intended for internal use cases until we know what we need.
        API will need to change to handle models with two exog.
        This is not yet supported by all model subclasses.

        This is essentially a simplified version of `fit_constrained`, and
        does not need to use `offset`.

        The estimation creates a new model with transformed design matrix,
        exog, and converts the results back to the original parameterization.

        Some subclasses could use a more efficient calculation than using a
        new model.

        Parameters
        ----------
        keep_index : array_like (int or bool) or slice
            variables that should be dropped.
        start_params : None or array_like
            starting values for the optimization. `start_params` needs to be
            given in the original parameter space and are internally
            transformed.
        k_params : int or None
            If None, then we try to infer from start_params or model.
        **fit_kwds : keyword arguments
            fit_kwds are used in the optimization of the transformed model.

        Returns
        -------
        results : Results instance
        """
    if hasattr(self, 'k_extra') and self.k_extra > 0:
        keep_index = np.array(keep_index, copy=True)
        k = self.exog.shape[1]
        extra_index = np.arange(k, k + self.k_extra)
        keep_index_p = np.concatenate((keep_index, extra_index))
    else:
        keep_index_p = keep_index
    if start_params is not None:
        fit_kwds['start_params'] = start_params[keep_index_p]
        k_params = len(start_params)
    init_kwds = self._get_init_kwds()
    mod_constr = self.__class__(self.endog, self.exog[:, keep_index], **init_kwds)
    res_constr = mod_constr.fit(**fit_kwds)
    keep_index = keep_index_p
    if k_params is None:
        k_params = self.exog.shape[1]
        k_params += getattr(self, 'k_extra', 0)
    params_full = np.zeros(k_params)
    params_full[keep_index] = res_constr.params
    try:
        res = self.fit(maxiter=0, disp=0, method='nm', skip_hessian=True, warn_convergence=False, start_params=params_full)
    except (TypeError, ValueError):
        res = self.fit()
    if hasattr(res_constr.model, 'scale'):
        res.model.scale = res._results.scale = res_constr.model.scale
    if hasattr(res_constr, 'mle_retvals'):
        res._results.mle_retvals = res_constr.mle_retvals
    if hasattr(res_constr, 'mle_settings'):
        res._results.mle_settings = res_constr.mle_settings
    res._results.params = params_full
    if not hasattr(res._results, 'normalized_cov_params') or res._results.normalized_cov_params is None:
        res._results.normalized_cov_params = np.zeros((k_params, k_params))
    else:
        res._results.normalized_cov_params[...] = 0
    keep_index = np.array(keep_index)
    res._results.normalized_cov_params[keep_index[:, None], keep_index] = res_constr.normalized_cov_params
    k_constr = res_constr.df_resid - res._results.df_resid
    if hasattr(res_constr, 'cov_params_default'):
        res._results.cov_params_default = np.zeros((k_params, k_params))
        res._results.cov_params_default[keep_index[:, None], keep_index] = res_constr.cov_params_default
    if hasattr(res_constr, 'cov_type'):
        res._results.cov_type = res_constr.cov_type
        res._results.cov_kwds = res_constr.cov_kwds
    res._results.keep_index = keep_index
    res._results.df_resid = res_constr.df_resid
    res._results.df_model = res_constr.df_model
    res._results.k_constr = k_constr
    res._results.results_constrained = res_constr
    if hasattr(res.model, 'M'):
        del res._results._cache['resid']
        del res._results._cache['fittedvalues']
        del res._results._cache['sresid']
        cov = res._results._cache['bcov_scaled']
        cov[...] = 0
        cov[keep_index[:, None], keep_index] = res_constr.bcov_scaled
        res._results.cov_params_default = cov
    return res