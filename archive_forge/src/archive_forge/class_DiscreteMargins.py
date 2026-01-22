from statsmodels.compat.python import lzip
import numpy as np
from scipy.stats import norm
from statsmodels.tools.decorators import cache_readonly
class DiscreteMargins:
    """Get marginal effects of a Discrete Choice model.

    Parameters
    ----------
    results : DiscreteResults instance
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

        Notes
        -----
        The dataframe is created on each call and not cached, as are the
        tables build in `summary()`
        """
        _check_at_is_all(self.margeff_options)
        results = self.results
        model = self.results.model
        from pandas import DataFrame, MultiIndex
        names = [_transform_names[self.margeff_options['method']], 'Std. Err.', 'z', 'Pr(>|z|)', 'Conf. Int. Low', 'Cont. Int. Hi.']
        ind = self.results.model.exog.var(0) != 0
        exog_names = self.results.model.exog_names
        k_extra = getattr(model, 'k_extra', 0)
        if k_extra > 0:
            exog_names = exog_names[:-k_extra]
        var_names = [name for i, name in enumerate(exog_names) if ind[i]]
        if self.margeff.ndim == 2:
            ci = self.conf_int(alpha)
            table = np.column_stack([i.ravel('F') for i in [self.margeff, self.margeff_se, self.tvalues, self.pvalues, ci[:, 0, :], ci[:, 1, :]]])
            _, yname_list = results._get_endog_name(model.endog_names, None, all=True)
            ynames = np.repeat(yname_list, len(var_names))
            xnames = np.tile(var_names, len(yname_list))
            index = MultiIndex.from_tuples(list(zip(ynames, xnames)), names=['endog', 'exog'])
        else:
            table = np.column_stack((self.margeff, self.margeff_se, self.tvalues, self.pvalues, self.conf_int(alpha)))
            index = var_names
        return DataFrame(table, columns=names, index=index)

    @cache_readonly
    def pvalues(self):
        _check_at_is_all(self.margeff_options)
        return norm.sf(np.abs(self.tvalues)) * 2

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
        q = norm.ppf(1 - alpha / 2)
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
        _, const_idx = _get_const_index(model.exog)
        if const_idx is not None:
            exog_names.pop(const_idx[0])
        if getattr(model, 'k_extra', 0) > 0:
            exog_names = exog_names[:-model.k_extra]
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
                header = ['', _transform_names[method], 'std err', 'z', 'P>|z|', '[' + str(alpha / 2), str(1 - alpha / 2) + ']']
                tble.insert_header_row(0, header)
                table.append(tble)
            table = table_extend(table, keep_headers=True)
        else:
            restup = (results, margeff, margeff_se, tvalues, pvalues, conf_int)
            table = summary_params(restup, yname=yname, xname=exog_names, alpha=alpha, use_t=False, skip_header=True)
            header = ['', _transform_names[method], 'std err', 'z', 'P>|z|', '[' + str(alpha / 2), str(1 - alpha / 2) + ']']
            table.insert_header_row(0, header)
        smry.tables.append(table)
        return smry

    def get_margeff(self, at='overall', method='dydx', atexog=None, dummy=False, count=False):
        """Get marginal effects of the fitted model.

        Parameters
        ----------
        at : str, optional
            Options are:

            - 'overall', The average of the marginal effects at each
              observation.
            - 'mean', The marginal effects at the mean of each regressor.
            - 'median', The marginal effects at the median of each regressor.
            - 'zero', The marginal effects at zero for each regressor.
            - 'all', The marginal effects at each observation. If `at` is all
              only margeff will be available.

            Note that if `exog` is specified, then marginal effects for all
            variables not specified by `exog` are calculated using the `at`
            option.
        method : str, optional
            Options are:

            - 'dydx' - dy/dx - No transformation is made and marginal effects
              are returned.  This is the default.
            - 'eyex' - estimate elasticities of variables in `exog` --
              d(lny)/d(lnx)
            - 'dyex' - estimate semi-elasticity -- dy/d(lnx)
            - 'eydx' - estimate semi-elasticity -- d(lny)/dx

            Note that tranformations are done after each observation is
            calculated.  Semi-elasticities for binary variables are computed
            using the midpoint method. 'dyex' and 'eyex' do not make sense
            for discrete variables.
        atexog : array_like, optional
            Optionally, you can provide the exogenous variables over which to
            get the marginal effects.  This should be a dictionary with the key
            as the zero-indexed column number and the value of the dictionary.
            Default is None for all independent variables less the constant.
        dummy : bool, optional
            If False, treats binary variables (if present) as continuous.  This
            is the default.  Else if True, treats binary variables as
            changing from 0 to 1.  Note that any variable that is either 0 or 1
            is treated as binary.  Each binary variable is treated separately
            for now.
        count : bool, optional
            If False, treats count variables (if present) as continuous.  This
            is the default.  Else if True, the marginal effect is the
            change in probabilities when each observation is increased by one.

        Returns
        -------
        effects : ndarray
            the marginal effect corresponding to the input options

        Notes
        -----
        When using after Poisson, returns the expected number of events
        per period, assuming that the model is loglinear.
        """
        self._reset()
        method = method.lower()
        at = at.lower()
        _check_margeff_args(at, method)
        self.margeff_options = dict(method=method, at=at)
        results = self.results
        model = results.model
        params = results.params
        exog = model.exog.copy()
        effects_idx, const_idx = _get_const_index(exog)
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
        self.dummy_idx = dummy_idx
        self.count_idx = count_idx
        exog = _get_margeff_exog(exog, at, atexog, effects_idx)
        effects = model._derivative_exog(params, exog, method, dummy_idx, count_idx)
        J = getattr(model, 'J', 1)
        effects_idx = np.tile(effects_idx, J)
        effects = _effects_at(effects, at)
        if at == 'all':
            if J > 1:
                K = model.K - np.any(~effects_idx)
                self.margeff = effects[:, effects_idx].reshape(-1, K, J, order='F')
            else:
                self.margeff = effects[:, effects_idx]
        else:
            margeff_cov, margeff_se = margeff_cov_with_se(model, params, exog, results.cov_params(), at, model._derivative_exog, dummy_idx, count_idx, method, J)
            if J > 1:
                K = model.K - np.any(~effects_idx)
                self.margeff = effects[effects_idx].reshape(K, J, order='F')
                self.margeff_se = margeff_se[effects_idx].reshape(K, J, order='F')
                self.margeff_cov = margeff_cov[effects_idx][:, effects_idx]
            else:
                effects_idx = effects_idx[:len(effects)]
                self.margeff_cov = margeff_cov[effects_idx][:, effects_idx]
                self.margeff_se = margeff_se[effects_idx]
                self.margeff = effects[effects_idx]