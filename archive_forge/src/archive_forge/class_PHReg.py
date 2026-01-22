import numpy as np
from statsmodels.base import model
import statsmodels.base.model as base
from statsmodels.tools.decorators import cache_readonly
from statsmodels.compat.pandas import Appender
class PHReg(model.LikelihoodModel):
    """
    Cox Proportional Hazards Regression Model

    The Cox PH Model is for right censored data.

    Parameters
    ----------
    endog : array_like
        The observed times (event or censoring)
    exog : 2D array_like
        The covariates or exogeneous variables
    status : array_like
        The censoring status values; status=1 indicates that an
        event occurred (e.g. failure or death), status=0 indicates
        that the observation was right censored. If None, defaults
        to status=1 for all cases.
    entry : array_like
        The entry times, if left truncation occurs
    strata : array_like
        Stratum labels.  If None, all observations are taken to be
        in a single stratum.
    ties : str
        The method used to handle tied times, must be either 'breslow'
        or 'efron'.
    offset : array_like
        Array of offset values
    missing : str
        The method used to handle missing data

    Notes
    -----
    Proportional hazards regression models should not include an
    explicit or implicit intercept.  The effect of an intercept is
    not identified using the partial likelihood approach.

    `endog`, `event`, `strata`, `entry`, and the first dimension
    of `exog` all must have the same length
    """

    def __init__(self, endog, exog, status=None, entry=None, strata=None, offset=None, ties='breslow', missing='drop', **kwargs):
        if status is None:
            status = np.ones(len(endog))
        super().__init__(endog, exog, status=status, entry=entry, strata=strata, offset=offset, missing=missing, **kwargs)
        if self.status is not None:
            self.status = np.asarray(self.status)
        if self.entry is not None:
            self.entry = np.asarray(self.entry)
        if self.strata is not None:
            self.strata = np.asarray(self.strata)
        if self.offset is not None:
            self.offset = np.asarray(self.offset)
        self.surv = PHSurvivalTime(self.endog, self.status, self.exog, self.strata, self.entry, self.offset)
        self.nobs = len(self.endog)
        self.groups = None
        self.missing = missing
        self.df_resid = float(self.exog.shape[0] - np.linalg.matrix_rank(self.exog))
        self.df_model = float(np.linalg.matrix_rank(self.exog))
        ties = ties.lower()
        if ties not in ('efron', 'breslow'):
            raise ValueError('`ties` must be either `efron` or ' + '`breslow`')
        self.ties = ties

    @classmethod
    def from_formula(cls, formula, data, status=None, entry=None, strata=None, offset=None, subset=None, ties='breslow', missing='drop', *args, **kwargs):
        """
        Create a proportional hazards regression model from a formula
        and dataframe.

        Parameters
        ----------
        formula : str or generic Formula object
            The formula specifying the model
        data : array_like
            The data for the model. See Notes.
        status : array_like
            The censoring status values; status=1 indicates that an
            event occurred (e.g. failure or death), status=0 indicates
            that the observation was right censored. If None, defaults
            to status=1 for all cases.
        entry : array_like
            The entry times, if left truncation occurs
        strata : array_like
            Stratum labels.  If None, all observations are taken to be
            in a single stratum.
        offset : array_like
            Array of offset values
        subset : array_like
            An array-like object of booleans, integers, or index
            values that indicate the subset of df to use in the
            model. Assumes df is a `pandas.DataFrame`
        ties : str
            The method used to handle tied times, must be either 'breslow'
            or 'efron'.
        missing : str
            The method used to handle missing data
        args : extra arguments
            These are passed to the model
        kwargs : extra keyword arguments
            These are passed to the model with one exception. The
            ``eval_env`` keyword is passed to patsy. It can be either a
            :class:`patsy:patsy.EvalEnvironment` object or an integer
            indicating the depth of the namespace to use. For example, the
            default ``eval_env=0`` uses the calling namespace. If you wish
            to use a "clean" environment set ``eval_env=-1``.

        Returns
        -------
        model : PHReg model instance
        """
        if isinstance(status, str):
            status = data[status]
        if isinstance(entry, str):
            entry = data[entry]
        if isinstance(strata, str):
            strata = data[strata]
        if isinstance(offset, str):
            offset = data[offset]
        import re
        terms = re.split('[+\\-~]', formula)
        for term in terms:
            term = term.strip()
            if term in ('0', '1'):
                import warnings
                warnings.warn("PHReg formulas should not include any '0' or '1' terms")
        mod = super().from_formula(formula, data, *args, status=status, entry=entry, strata=strata, offset=offset, subset=subset, ties=ties, missing=missing, drop_cols=['Intercept'], **kwargs)
        return mod

    def fit(self, groups=None, **args):
        """
        Fit a proportional hazards regression model.

        Parameters
        ----------
        groups : array_like
            Labels indicating groups of observations that may be
            dependent.  If present, the standard errors account for
            this dependence. Does not affect fitted values.

        Returns
        -------
        PHRegResults
            Returns a results instance.
        """
        if groups is not None:
            if len(groups) != len(self.endog):
                msg = 'len(groups) = %d and len(endog) = %d differ' % (len(groups), len(self.endog))
                raise ValueError(msg)
            self.groups = np.asarray(groups)
        else:
            self.groups = None
        if 'disp' not in args:
            args['disp'] = False
        fit_rslts = super().fit(**args)
        if self.groups is None:
            cov_params = fit_rslts.cov_params()
        else:
            cov_params = self.robust_covariance(fit_rslts.params)
        results = PHRegResults(self, fit_rslts.params, cov_params)
        return results

    def fit_regularized(self, method='elastic_net', alpha=0.0, start_params=None, refit=False, **kwargs):
        """
        Return a regularized fit to a linear regression model.

        Parameters
        ----------
        method : {'elastic_net'}
            Only the `elastic_net` approach is currently implemented.
        alpha : scalar or array_like
            The penalty weight.  If a scalar, the same penalty weight
            applies to all variables in the model.  If a vector, it
            must have the same length as `params`, and contains a
            penalty weight for each coefficient.
        start_params : array_like
            Starting values for `params`.
        refit : bool
            If True, the model is refit using only the variables that
            have non-zero coefficients in the regularized fit.  The
            refitted model is not regularized.
        **kwargs
            Additional keyword arguments used to fit the model.

        Returns
        -------
        PHRegResults
            Returns a results instance.

        Notes
        -----
        The penalty is the ``elastic net`` penalty, which is a
        combination of L1 and L2 penalties.

        The function that is minimized is:

        .. math::

            -loglike/n + alpha*((1-L1\\_wt)*|params|_2^2/2 + L1\\_wt*|params|_1)

        where :math:`|*|_1` and :math:`|*|_2` are the L1 and L2 norms.

        Post-estimation results are based on the same data used to
        select variables, hence may be subject to overfitting biases.

        The elastic_net method uses the following keyword arguments:

        maxiter : int
            Maximum number of iterations
        L1_wt  : float
            Must be in [0, 1].  The L1 penalty has weight L1_wt and the
            L2 penalty has weight 1 - L1_wt.
        cnvrg_tol : float
            Convergence threshold for line searches
        zero_tol : float
            Coefficients below this threshold are treated as zero.
        """
        from statsmodels.base.elastic_net import fit_elasticnet
        if method != 'elastic_net':
            raise ValueError('method for fit_regularized must be elastic_net')
        defaults = {'maxiter': 50, 'L1_wt': 1, 'cnvrg_tol': 1e-10, 'zero_tol': 1e-10}
        defaults.update(kwargs)
        return fit_elasticnet(self, method=method, alpha=alpha, start_params=start_params, refit=refit, **defaults)

    def loglike(self, params):
        """
        Returns the log partial likelihood function evaluated at
        `params`.
        """
        if self.ties == 'breslow':
            return self.breslow_loglike(params)
        elif self.ties == 'efron':
            return self.efron_loglike(params)

    def score(self, params):
        """
        Returns the score function evaluated at `params`.
        """
        if self.ties == 'breslow':
            return self.breslow_gradient(params)
        elif self.ties == 'efron':
            return self.efron_gradient(params)

    def hessian(self, params):
        """
        Returns the Hessian matrix of the log partial likelihood
        function evaluated at `params`.
        """
        if self.ties == 'breslow':
            return self.breslow_hessian(params)
        else:
            return self.efron_hessian(params)

    def breslow_loglike(self, params):
        """
        Returns the value of the log partial likelihood function
        evaluated at `params`, using the Breslow method to handle tied
        times.
        """
        surv = self.surv
        like = 0.0
        for stx in range(surv.nstrat):
            uft_ix = surv.ufailt_ix[stx]
            exog_s = surv.exog_s[stx]
            nuft = len(uft_ix)
            linpred = np.dot(exog_s, params)
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)
            xp0 = 0.0
            for i in range(nuft)[::-1]:
                ix = surv.risk_enter[stx][i]
                xp0 += e_linpred[ix].sum()
                ix = uft_ix[i]
                like += (linpred[ix] - np.log(xp0)).sum()
                ix = surv.risk_exit[stx][i]
                xp0 -= e_linpred[ix].sum()
        return like

    def efron_loglike(self, params):
        """
        Returns the value of the log partial likelihood function
        evaluated at `params`, using the Efron method to handle tied
        times.
        """
        surv = self.surv
        like = 0.0
        for stx in range(surv.nstrat):
            exog_s = surv.exog_s[stx]
            linpred = np.dot(exog_s, params)
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)
            xp0 = 0.0
            uft_ix = surv.ufailt_ix[stx]
            nuft = len(uft_ix)
            for i in range(nuft)[::-1]:
                ix = surv.risk_enter[stx][i]
                xp0 += e_linpred[ix].sum()
                xp0f = e_linpred[uft_ix[i]].sum()
                ix = uft_ix[i]
                like += linpred[ix].sum()
                m = len(ix)
                J = np.arange(m, dtype=np.float64) / m
                like -= np.log(xp0 - J * xp0f).sum()
                ix = surv.risk_exit[stx][i]
                xp0 -= e_linpred[ix].sum()
        return like

    def breslow_gradient(self, params):
        """
        Returns the gradient of the log partial likelihood, using the
        Breslow method to handle tied times.
        """
        surv = self.surv
        grad = 0.0
        for stx in range(surv.nstrat):
            strat_ix = surv.stratum_rows[stx]
            uft_ix = surv.ufailt_ix[stx]
            nuft = len(uft_ix)
            exog_s = surv.exog_s[stx]
            linpred = np.dot(exog_s, params)
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)
            xp0, xp1 = (0.0, 0.0)
            for i in range(nuft)[::-1]:
                ix = surv.risk_enter[stx][i]
                if len(ix) > 0:
                    v = exog_s[ix, :]
                    xp0 += e_linpred[ix].sum()
                    xp1 += (e_linpred[ix][:, None] * v).sum(0)
                ix = uft_ix[i]
                grad += (exog_s[ix, :] - xp1 / xp0).sum(0)
                ix = surv.risk_exit[stx][i]
                if len(ix) > 0:
                    v = exog_s[ix, :]
                    xp0 -= e_linpred[ix].sum()
                    xp1 -= (e_linpred[ix][:, None] * v).sum(0)
        return grad

    def efron_gradient(self, params):
        """
        Returns the gradient of the log partial likelihood evaluated
        at `params`, using the Efron method to handle tied times.
        """
        surv = self.surv
        grad = 0.0
        for stx in range(surv.nstrat):
            strat_ix = surv.stratum_rows[stx]
            exog_s = surv.exog_s[stx]
            linpred = np.dot(exog_s, params)
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)
            xp0, xp1 = (0.0, 0.0)
            uft_ix = surv.ufailt_ix[stx]
            nuft = len(uft_ix)
            for i in range(nuft)[::-1]:
                ix = surv.risk_enter[stx][i]
                if len(ix) > 0:
                    v = exog_s[ix, :]
                    xp0 += e_linpred[ix].sum()
                    xp1 += (e_linpred[ix][:, None] * v).sum(0)
                ixf = uft_ix[i]
                if len(ixf) > 0:
                    v = exog_s[ixf, :]
                    xp0f = e_linpred[ixf].sum()
                    xp1f = (e_linpred[ixf][:, None] * v).sum(0)
                    grad += v.sum(0)
                    m = len(ixf)
                    J = np.arange(m, dtype=np.float64) / m
                    numer = xp1 - np.outer(J, xp1f)
                    denom = xp0 - np.outer(J, xp0f)
                    ratio = numer / denom
                    rsum = ratio.sum(0)
                    grad -= rsum
                ix = surv.risk_exit[stx][i]
                if len(ix) > 0:
                    v = exog_s[ix, :]
                    xp0 -= e_linpred[ix].sum()
                    xp1 -= (e_linpred[ix][:, None] * v).sum(0)
        return grad

    def breslow_hessian(self, params):
        """
        Returns the Hessian of the log partial likelihood evaluated at
        `params`, using the Breslow method to handle tied times.
        """
        surv = self.surv
        hess = 0.0
        for stx in range(surv.nstrat):
            uft_ix = surv.ufailt_ix[stx]
            nuft = len(uft_ix)
            exog_s = surv.exog_s[stx]
            linpred = np.dot(exog_s, params)
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)
            xp0, xp1, xp2 = (0.0, 0.0, 0.0)
            for i in range(nuft)[::-1]:
                ix = surv.risk_enter[stx][i]
                if len(ix) > 0:
                    xp0 += e_linpred[ix].sum()
                    v = exog_s[ix, :]
                    xp1 += (e_linpred[ix][:, None] * v).sum(0)
                    elx = e_linpred[ix]
                    xp2 += np.einsum('ij,ik,i->jk', v, v, elx)
                m = len(uft_ix[i])
                hess += m * (xp2 / xp0 - np.outer(xp1, xp1) / xp0 ** 2)
                ix = surv.risk_exit[stx][i]
                if len(ix) > 0:
                    xp0 -= e_linpred[ix].sum()
                    v = exog_s[ix, :]
                    xp1 -= (e_linpred[ix][:, None] * v).sum(0)
                    elx = e_linpred[ix]
                    xp2 -= np.einsum('ij,ik,i->jk', v, v, elx)
        return -hess

    def efron_hessian(self, params):
        """
        Returns the Hessian matrix of the partial log-likelihood
        evaluated at `params`, using the Efron method to handle tied
        times.
        """
        surv = self.surv
        hess = 0.0
        for stx in range(surv.nstrat):
            exog_s = surv.exog_s[stx]
            linpred = np.dot(exog_s, params)
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)
            xp0, xp1, xp2 = (0.0, 0.0, 0.0)
            uft_ix = surv.ufailt_ix[stx]
            nuft = len(uft_ix)
            for i in range(nuft)[::-1]:
                ix = surv.risk_enter[stx][i]
                if len(ix) > 0:
                    xp0 += e_linpred[ix].sum()
                    v = exog_s[ix, :]
                    xp1 += (e_linpred[ix][:, None] * v).sum(0)
                    elx = e_linpred[ix]
                    xp2 += np.einsum('ij,ik,i->jk', v, v, elx)
                ixf = uft_ix[i]
                if len(ixf) > 0:
                    v = exog_s[ixf, :]
                    xp0f = e_linpred[ixf].sum()
                    xp1f = (e_linpred[ixf][:, None] * v).sum(0)
                    elx = e_linpred[ixf]
                    xp2f = np.einsum('ij,ik,i->jk', v, v, elx)
                m = len(uft_ix[i])
                J = np.arange(m, dtype=np.float64) / m
                c0 = xp0 - J * xp0f
                hess += xp2 * np.sum(1 / c0)
                hess -= xp2f * np.sum(J / c0)
                mat = (xp1[None, :] - np.outer(J, xp1f)) / c0[:, None]
                hess -= np.einsum('ij,ik->jk', mat, mat)
                ix = surv.risk_exit[stx][i]
                if len(ix) > 0:
                    xp0 -= e_linpred[ix].sum()
                    v = exog_s[ix, :]
                    xp1 -= (e_linpred[ix][:, None] * v).sum(0)
                    elx = e_linpred[ix]
                    xp2 -= np.einsum('ij,ik,i->jk', v, v, elx)
        return -hess

    def robust_covariance(self, params):
        """
        Returns a covariance matrix for the proportional hazards model
        regresion coefficient estimates that is robust to certain
        forms of model misspecification.

        Parameters
        ----------
        params : ndarray
            The parameter vector at which the covariance matrix is
            calculated.

        Returns
        -------
        The robust covariance matrix as a square ndarray.

        Notes
        -----
        This function uses the `groups` argument to determine groups
        within which observations may be dependent.  The covariance
        matrix is calculated using the Huber-White "sandwich" approach.
        """
        if self.groups is None:
            raise ValueError('`groups` must be specified to calculate the robust covariance matrix')
        hess = self.hessian(params)
        score_obs = self.score_residuals(params)
        grads = {}
        for i, g in enumerate(self.groups):
            if g not in grads:
                grads[g] = 0.0
            grads[g] += score_obs[i, :]
        grads = np.asarray(list(grads.values()))
        mat = grads[None, :, :]
        mat = mat.T * mat
        mat = mat.sum(1)
        hess_inv = np.linalg.inv(hess)
        cmat = np.dot(hess_inv, np.dot(mat, hess_inv))
        return cmat

    def score_residuals(self, params):
        """
        Returns the score residuals calculated at a given vector of
        parameters.

        Parameters
        ----------
        params : ndarray
            The parameter vector at which the score residuals are
            calculated.

        Returns
        -------
        The score residuals, returned as a ndarray having the same
        shape as `exog`.

        Notes
        -----
        Observations in a stratum with no observed events have undefined
        score residuals, and contain NaN in the returned matrix.
        """
        surv = self.surv
        score_resid = np.zeros(self.exog.shape, dtype=np.float64)
        mask = np.zeros(self.exog.shape[0], dtype=np.int32)
        w_avg = self.weighted_covariate_averages(params)
        for stx in range(surv.nstrat):
            uft_ix = surv.ufailt_ix[stx]
            exog_s = surv.exog_s[stx]
            nuft = len(uft_ix)
            strat_ix = surv.stratum_rows[stx]
            xp0 = 0.0
            linpred = np.dot(exog_s, params)
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)
            at_risk_ix = set()
            for i in range(nuft)[::-1]:
                ix = surv.risk_enter[stx][i]
                at_risk_ix |= set(ix)
                xp0 += e_linpred[ix].sum()
                atr_ix = list(at_risk_ix)
                leverage = exog_s[atr_ix, :] - w_avg[stx][i, :]
                d = np.zeros(exog_s.shape[0])
                d[uft_ix[i]] = 1
                dchaz = len(uft_ix[i]) / xp0
                mrp = d[atr_ix] - e_linpred[atr_ix] * dchaz
                ii = strat_ix[atr_ix]
                score_resid[ii, :] += leverage * mrp[:, None]
                mask[ii] = 1
                ix = surv.risk_exit[stx][i]
                at_risk_ix -= set(ix)
                xp0 -= e_linpred[ix].sum()
        jj = np.flatnonzero(mask == 0)
        if len(jj) > 0:
            score_resid[jj, :] = np.nan
        return score_resid

    def weighted_covariate_averages(self, params):
        """
        Returns the hazard-weighted average of covariate values for
        subjects who are at-risk at a particular time.

        Parameters
        ----------
        params : ndarray
            Parameter vector

        Returns
        -------
        averages : list of ndarrays
            averages[stx][i,:] is a row vector containing the weighted
            average values (for all the covariates) of at-risk
            subjects a the i^th largest observed failure time in
            stratum `stx`, using the hazard multipliers as weights.

        Notes
        -----
        Used to calculate leverages and score residuals.
        """
        surv = self.surv
        averages = []
        xp0, xp1 = (0.0, 0.0)
        for stx in range(surv.nstrat):
            uft_ix = surv.ufailt_ix[stx]
            exog_s = surv.exog_s[stx]
            nuft = len(uft_ix)
            average_s = np.zeros((len(uft_ix), exog_s.shape[1]), dtype=np.float64)
            linpred = np.dot(exog_s, params)
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)
            for i in range(nuft)[::-1]:
                ix = surv.risk_enter[stx][i]
                xp0 += e_linpred[ix].sum()
                xp1 += np.dot(e_linpred[ix], exog_s[ix, :])
                average_s[i, :] = xp1 / xp0
                ix = surv.risk_exit[stx][i]
                xp0 -= e_linpred[ix].sum()
                xp1 -= np.dot(e_linpred[ix], exog_s[ix, :])
            averages.append(average_s)
        return averages

    def baseline_cumulative_hazard(self, params):
        """
        Estimate the baseline cumulative hazard and survival
        functions.

        Parameters
        ----------
        params : ndarray
            The model parameters.

        Returns
        -------
        A list of triples (time, hazard, survival) containing the time
        values and corresponding cumulative hazard and survival
        function values for each stratum.

        Notes
        -----
        Uses the Nelson-Aalen estimator.
        """
        surv = self.surv
        rslt = []
        for stx in range(surv.nstrat):
            uft = surv.ufailt[stx]
            uft_ix = surv.ufailt_ix[stx]
            exog_s = surv.exog_s[stx]
            nuft = len(uft_ix)
            linpred = np.dot(exog_s, params)
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            e_linpred = np.exp(linpred)
            xp0 = 0.0
            h0 = np.zeros(nuft, dtype=np.float64)
            for i in range(nuft)[::-1]:
                ix = surv.risk_enter[stx][i]
                xp0 += e_linpred[ix].sum()
                ix = uft_ix[i]
                h0[i] = len(ix) / xp0
                ix = surv.risk_exit[stx][i]
                xp0 -= e_linpred[ix].sum()
            cumhaz = np.cumsum(h0) - h0
            current_strata_surv = np.exp(-cumhaz)
            rslt.append([uft, cumhaz, current_strata_surv])
        return rslt

    def baseline_cumulative_hazard_function(self, params):
        """
        Returns a function that calculates the baseline cumulative
        hazard function for each stratum.

        Parameters
        ----------
        params : ndarray
            The model parameters.

        Returns
        -------
        A dict mapping stratum names to the estimated baseline
        cumulative hazard function.
        """
        from scipy.interpolate import interp1d
        surv = self.surv
        base = self.baseline_cumulative_hazard(params)
        cumhaz_f = {}
        for stx in range(surv.nstrat):
            time_h = base[stx][0]
            cumhaz = base[stx][1]
            time_h = np.r_[-np.inf, time_h, np.inf]
            cumhaz = np.r_[cumhaz[0], cumhaz, cumhaz[-1]]
            func = interp1d(time_h, cumhaz, kind='zero')
            cumhaz_f[self.surv.stratum_names[stx]] = func
        return cumhaz_f

    @Appender(_predict_docstring % {'params_doc': _predict_params_doc, 'cov_params_doc': _predict_cov_params_docstring})
    def predict(self, params, exog=None, cov_params=None, endog=None, strata=None, offset=None, pred_type='lhr', pred_only=False):
        pred_type = pred_type.lower()
        if pred_type not in ['lhr', 'hr', 'surv', 'cumhaz']:
            msg = 'Type %s not allowed for prediction' % pred_type
            raise ValueError(msg)

        class bunch:
            predicted_values = None
            standard_errors = None
        ret_val = bunch()
        exog_provided = True
        if exog is None:
            exog = self.exog
            exog_provided = False
        lhr = np.dot(exog, params)
        if offset is not None:
            lhr += offset
        elif self.offset is not None and (not exog_provided):
            lhr += self.offset
        if pred_type == 'lhr':
            ret_val.predicted_values = lhr
            if cov_params is not None:
                mat = np.dot(exog, cov_params)
                va = (mat * exog).sum(1)
                ret_val.standard_errors = np.sqrt(va)
            if pred_only:
                return ret_val.predicted_values
            return ret_val
        hr = np.exp(lhr)
        if pred_type == 'hr':
            ret_val.predicted_values = hr
            if pred_only:
                return ret_val.predicted_values
            return ret_val
        if endog is None and exog_provided:
            msg = 'If `exog` is provided `endog` must be provided.'
            raise ValueError(msg)
        elif endog is None and (not exog_provided):
            endog = self.endog
        if strata is None:
            if exog_provided and self.surv.nstrat > 1:
                raise ValueError('`strata` must be provided')
            if self.strata is None:
                strata = [self.surv.stratum_names[0]] * len(endog)
            else:
                strata = self.strata
        cumhaz = np.nan * np.ones(len(endog), dtype=np.float64)
        stv = np.unique(strata)
        bhaz = self.baseline_cumulative_hazard_function(params)
        for stx in stv:
            ix = np.flatnonzero(strata == stx)
            func = bhaz[stx]
            cumhaz[ix] = func(endog[ix]) * hr[ix]
        if pred_type == 'cumhaz':
            ret_val.predicted_values = cumhaz
        elif pred_type == 'surv':
            ret_val.predicted_values = np.exp(-cumhaz)
        if pred_only:
            return ret_val.predicted_values
        return ret_val

    def get_distribution(self, params, scale=1.0, exog=None):
        """
        Returns a scipy distribution object corresponding to the
        distribution of uncensored endog (duration) values for each
        case.

        Parameters
        ----------
        params : array_like
            The proportional hazards model parameters.
        scale : float
            Present for compatibility, not used.
        exog : array_like
            A design matrix, defaults to model.exog.

        Returns
        -------
        A list of objects of type scipy.stats.distributions.rv_discrete

        Notes
        -----
        The distributions are obtained from a simple discrete estimate
        of the survivor function that puts all mass on the observed
        failure times within a stratum.
        """
        surv = self.surv
        bhaz = self.baseline_cumulative_hazard(params)
        pk, xk = ([], [])
        if exog is None:
            exog_split = surv.exog_s
        else:
            exog_split = self.surv._split(exog)
        for stx in range(self.surv.nstrat):
            exog_s = exog_split[stx]
            linpred = np.dot(exog_s, params)
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            e_linpred = np.exp(linpred)
            pts = bhaz[stx][0]
            ichaz = np.outer(e_linpred, bhaz[stx][1])
            usurv = np.exp(-ichaz)
            z = np.zeros((usurv.shape[0], 1))
            usurv = np.concatenate((usurv, z), axis=1)
            probs = -np.diff(usurv, 1)
            pk.append(probs)
            xk.append(np.outer(np.ones(probs.shape[0]), pts))
        mxc = max([x.shape[1] for x in xk])
        for k in range(self.surv.nstrat):
            if xk[k].shape[1] < mxc:
                xk1 = np.zeros((xk[k].shape[0], mxc))
                pk1 = np.zeros((pk[k].shape[0], mxc))
                xk1[:, 0:xk[k].shape[1]] = xk[k]
                pk1[:, 0:pk[k].shape[1]] = pk[k]
                xk[k], pk[k] = (xk1, pk1)
        xka = np.nan * np.ones((len(self.endog), mxc))
        pka = np.ones((len(self.endog), mxc), dtype=np.float64) / mxc
        for stx in range(self.surv.nstrat):
            ix = self.surv.stratum_rows[stx]
            xka[ix, :] = xk[stx]
            pka[ix, :] = pk[stx]
        dist = rv_discrete_float(xka, pka)
        return dist