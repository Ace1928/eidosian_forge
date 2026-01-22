from statsmodels.compat.python import lzip
from statsmodels.compat.pandas import Appender
import numpy as np
from scipy import stats
import pandas as pd
import patsy
from collections import defaultdict
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.model as base
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM, GLMResults
from statsmodels.genmod import cov_struct as cov_structs
import statsmodels.genmod.families.varfuncs as varfuncs
from statsmodels.genmod.families.links import Link
from statsmodels.tools.sm_exceptions import (ConvergenceWarning,
import warnings
from statsmodels.graphics._regressionplots_doc import (
from statsmodels.discrete.discrete_margins import (
class GEEMargins:
    """
    Estimated marginal effects for a regression model fit with GEE.

    Parameters
    ----------
    results : GEEResults instance
        The results instance of a fitted discrete choice model
    args : tuple
        Args are passed to `get_margeff`. This is the same as
        results.get_margeff. See there for more information.
    kwargs : dict
        Keyword args are passed to `get_margeff`. This is the same as
        results.get_margeff. See there for more information.
    """

    def __init__(self, results, args, kwargs={}):
        self._cache = {}
        self.results = results
        self.get_margeff(*args, **kwargs)

    def _reset(self):
        self._cache = {}

    @cache_readonly
    def tvalues(self):
        _check_at_is_all(self.margeff_options)
        return self.margeff / self.margeff_se

    def summary_frame(self, alpha=0.05):
        """
        Returns a DataFrame summarizing the marginal effects.

        Parameters
        ----------
        alpha : float
            Number between 0 and 1. The confidence intervals have the
            probability 1-alpha.

        Returns
        -------
        frame : DataFrames
            A DataFrame summarizing the marginal effects.
        """
        _check_at_is_all(self.margeff_options)
        from pandas import DataFrame
        names = [_transform_names[self.margeff_options['method']], 'Std. Err.', 'z', 'Pr(>|z|)', 'Conf. Int. Low', 'Cont. Int. Hi.']
        ind = self.results.model.exog.var(0) != 0
        exog_names = self.results.model.exog_names
        var_names = [name for i, name in enumerate(exog_names) if ind[i]]
        table = np.column_stack((self.margeff, self.margeff_se, self.tvalues, self.pvalues, self.conf_int(alpha)))
        return DataFrame(table, columns=names, index=var_names)

    @cache_readonly
    def pvalues(self):
        _check_at_is_all(self.margeff_options)
        return stats.norm.sf(np.abs(self.tvalues)) * 2

    def conf_int(self, alpha=0.05):
        """
        Returns the confidence intervals of the marginal effects

        Parameters
        ----------
        alpha : float
            Number between 0 and 1. The confidence intervals have the
            probability 1-alpha.

        Returns
        -------
        conf_int : ndarray
            An array with lower, upper confidence intervals for the marginal
            effects.
        """
        _check_at_is_all(self.margeff_options)
        me_se = self.margeff_se
        q = stats.norm.ppf(1 - alpha / 2)
        lower = self.margeff - q * me_se
        upper = self.margeff + q * me_se
        return np.asarray(lzip(lower, upper))

    def summary(self, alpha=0.05):
        """
        Returns a summary table for marginal effects

        Parameters
        ----------
        alpha : float
            Number between 0 and 1. The confidence intervals have the
            probability 1-alpha.

        Returns
        -------
        Summary : SummaryTable
            A SummaryTable instance
        """
        _check_at_is_all(self.margeff_options)
        results = self.results
        model = results.model
        title = model.__class__.__name__ + ' Marginal Effects'
        method = self.margeff_options['method']
        top_left = [('Dep. Variable:', [model.endog_names]), ('Method:', [method]), ('At:', [self.margeff_options['at']])]
        from statsmodels.iolib.summary import Summary, summary_params, table_extend
        exog_names = model.exog_names[:]
        smry = Summary()
        const_idx = model.data.const_idx
        if const_idx is not None:
            exog_names.pop(const_idx)
        J = int(getattr(model, 'J', 1))
        if J > 1:
            yname, yname_list = results._get_endog_name(model.endog_names, None, all=True)
        else:
            yname = model.endog_names
            yname_list = [yname]
        smry.add_table_2cols(self, gleft=top_left, gright=[], yname=yname, xname=exog_names, title=title)
        table = []
        conf_int = self.conf_int(alpha)
        margeff = self.margeff
        margeff_se = self.margeff_se
        tvalues = self.tvalues
        pvalues = self.pvalues
        if J > 1:
            for eq in range(J):
                restup = (results, margeff[:, eq], margeff_se[:, eq], tvalues[:, eq], pvalues[:, eq], conf_int[:, :, eq])
                tble = summary_params(restup, yname=yname_list[eq], xname=exog_names, alpha=alpha, use_t=False, skip_header=True)
                tble.title = yname_list[eq]
                header = ['', _transform_names[method], 'std err', 'z', 'P>|z|', '[%3.1f%% Conf. Int.]' % (100 - alpha * 100)]
                tble.insert_header_row(0, header)
                table.append(tble)
            table = table_extend(table, keep_headers=True)
        else:
            restup = (results, margeff, margeff_se, tvalues, pvalues, conf_int)
            table = summary_params(restup, yname=yname, xname=exog_names, alpha=alpha, use_t=False, skip_header=True)
            header = ['', _transform_names[method], 'std err', 'z', 'P>|z|', '[%3.1f%% Conf. Int.]' % (100 - alpha * 100)]
            table.insert_header_row(0, header)
        smry.tables.append(table)
        return smry

    def get_margeff(self, at='overall', method='dydx', atexog=None, dummy=False, count=False):
        self._reset()
        method = method.lower()
        at = at.lower()
        _check_margeff_args(at, method)
        self.margeff_options = dict(method=method, at=at)
        results = self.results
        model = results.model
        params = results.params
        exog = model.exog.copy()
        effects_idx = exog.var(0) != 0
        const_idx = model.data.const_idx
        if dummy:
            _check_discrete_args(at, method)
            dummy_idx, dummy = _get_dummy_index(exog, const_idx)
        else:
            dummy_idx = None
        if count:
            _check_discrete_args(at, method)
            count_idx, count = _get_count_index(exog, const_idx)
        else:
            count_idx = None
        exog = _get_margeff_exog(exog, at, atexog, effects_idx)
        effects = model._derivative_exog(params, exog, method, dummy_idx, count_idx)
        effects = _effects_at(effects, at)
        if at == 'all':
            self.margeff = effects[:, effects_idx]
        else:
            margeff_cov, margeff_se = margeff_cov_with_se(model, params, exog, results.cov_params(), at, model._derivative_exog, dummy_idx, count_idx, method, 1)
            self.margeff_cov = margeff_cov[effects_idx][:, effects_idx]
            self.margeff_se = margeff_se[effects_idx]
            self.margeff = effects[effects_idx]