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
class _LLRMixin:
    """Mixin class for Null model and likelihood ratio
    """

    def pseudo_rsquared(self, kind='mcf'):
        """
        McFadden's pseudo-R-squared. `1 - (llf / llnull)`
        """
        kind = kind.lower()
        if kind.startswith('mcf'):
            prsq = 1 - self.llf / self.llnull
        elif kind.startswith('cox') or kind in ['cs', 'lr']:
            prsq = 1 - np.exp((self.llnull - self.llf) * (2 / self.nobs))
        else:
            raise ValueError('only McFadden and Cox-Snell are available')
        return prsq

    @cache_readonly
    def llr(self):
        """
        Likelihood ratio chi-squared statistic; `-2*(llnull - llf)`
        """
        return -2 * (self.llnull - self.llf)

    @cache_readonly
    def llr_pvalue(self):
        """
        The chi-squared probability of getting a log-likelihood ratio
        statistic greater than llr.  llr has a chi-squared distribution
        with degrees of freedom `df_model`.
        """
        llr = self.llr
        df_full = self.df_resid
        df_restr = self.df_resid_null
        lrdf = df_restr - df_full
        self.df_lr_null = lrdf
        return stats.distributions.chi2.sf(llr, lrdf)

    def set_null_options(self, llnull=None, attach_results=True, **kwargs):
        """
        Set the fit options for the Null (constant-only) model.

        This resets the cache for related attributes which is potentially
        fragile. This only sets the option, the null model is estimated
        when llnull is accessed, if llnull is not yet in cache.

        Parameters
        ----------
        llnull : {None, float}
            If llnull is not None, then the value will be directly assigned to
            the cached attribute "llnull".
        attach_results : bool
            Sets an internal flag whether the results instance of the null
            model should be attached. By default without calling this method,
            thenull model results are not attached and only the loglikelihood
            value llnull is stored.
        **kwargs
            Additional keyword arguments used as fit keyword arguments for the
            null model. The override and model default values.

        Notes
        -----
        Modifies attributes of this instance, and so has no return.
        """
        self._cache.pop('llnull', None)
        self._cache.pop('llr', None)
        self._cache.pop('llr_pvalue', None)
        self._cache.pop('prsquared', None)
        if hasattr(self, 'res_null'):
            del self.res_null
        if llnull is not None:
            self._cache['llnull'] = llnull
        self._attach_nullmodel = attach_results
        self._optim_kwds_null = kwargs

    @cache_readonly
    def llnull(self):
        """
        Value of the constant-only loglikelihood
        """
        model = self.model
        kwds = model._get_init_kwds().copy()
        for key in getattr(model, '_null_drop_keys', []):
            del kwds[key]
        mod_null = model.__class__(model.endog, np.ones(self.nobs), **kwds)
        optim_kwds = getattr(self, '_optim_kwds_null', {}).copy()
        if 'start_params' in optim_kwds:
            sp_null = optim_kwds.pop('start_params')
        elif hasattr(model, '_get_start_params_null'):
            sp_null = model._get_start_params_null()
        else:
            sp_null = None
        opt_kwds = dict(method='bfgs', warn_convergence=False, maxiter=10000, disp=0)
        opt_kwds.update(optim_kwds)
        if optim_kwds:
            res_null = mod_null.fit(start_params=sp_null, **opt_kwds)
        else:
            res_null = mod_null.fit(start_params=sp_null, method='nm', warn_convergence=False, maxiter=10000, disp=0)
            res_null = mod_null.fit(start_params=res_null.params, method='bfgs', warn_convergence=False, maxiter=10000, disp=0)
        if getattr(self, '_attach_nullmodel', False) is not False:
            self.res_null = res_null
        self.k_null = len(res_null.params)
        self.df_resid_null = res_null.df_resid
        return res_null.llf