import numpy as np
import scipy.stats as stats
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
import statsmodels.regression._tools as reg_tools
import statsmodels.regression.linear_model as lm
import statsmodels.robust.norms as norms
import statsmodels.robust.scale as scale
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
class RLMResults(base.LikelihoodModelResults):
    """
    Class to contain RLM results

    Attributes
    ----------

    bcov_scaled : ndarray
        p x p scaled covariance matrix specified in the model fit method.
        The default is H1. H1 is defined as
        ``k**2 * (1/df_resid*sum(M.psi(sresid)**2)*scale**2)/
        ((1/nobs*sum(M.psi_deriv(sresid)))**2) * (X.T X)^(-1)``

        where ``k = 1 + (df_model +1)/nobs * var_psiprime/m**2``
        where ``m = mean(M.psi_deriv(sresid))`` and
        ``var_psiprime = var(M.psi_deriv(sresid))``

        H2 is defined as
        ``k * (1/df_resid) * sum(M.psi(sresid)**2) *scale**2/
        ((1/nobs)*sum(M.psi_deriv(sresid)))*W_inv``

        H3 is defined as
        ``1/k * (1/df_resid * sum(M.psi(sresid)**2)*scale**2 *
        (W_inv X.T X W_inv))``

        where `k` is defined as above and
        ``W_inv = (M.psi_deriv(sresid) exog.T exog)^(-1)``

        See the technical documentation for cleaner formulae.
    bcov_unscaled : ndarray
        The usual p x p covariance matrix with scale set equal to 1.  It
        is then just equivalent to normalized_cov_params.
    bse : ndarray
        An array of the standard errors of the parameters.  The standard
        errors are taken from the robust covariance matrix specified in the
        argument to fit.
    chisq : ndarray
        An array of the chi-squared values of the parameter estimates.
    df_model
        See RLM.df_model
    df_resid
        See RLM.df_resid
    fit_history : dict
        Contains information about the iterations. Its keys are `deviance`,
        `params`, `iteration` and the convergence criteria specified in
        `RLM.fit`, if different from `deviance` or `params`.
    fit_options : dict
        Contains the options given to fit.
    fittedvalues : ndarray
        The linear predicted values.  dot(exog, params)
    model : statsmodels.rlm.RLM
        A reference to the model instance
    nobs : float
        The number of observations n
    normalized_cov_params : ndarray
        See RLM.normalized_cov_params
    params : ndarray
        The coefficients of the fitted model
    pinv_wexog : ndarray
        See RLM.pinv_wexog
    pvalues : ndarray
        The p values associated with `tvalues`. Note that `tvalues` are assumed
        to be distributed standard normal rather than Student's t.
    resid : ndarray
        The residuals of the fitted model.  endog - fittedvalues
    scale : float
        The type of scale is determined in the arguments to the fit method in
        RLM.  The reported scale is taken from the residuals of the weighted
        least squares in the last IRLS iteration if update_scale is True.  If
        update_scale is False, then it is the scale given by the first OLS
        fit before the IRLS iterations.
    sresid : ndarray
        The scaled residuals.
    tvalues : ndarray
        The "t-statistics" of params. These are defined as params/bse where
        bse are taken from the robust covariance matrix specified in the
        argument to fit.
    weights : ndarray
        The reported weights are determined by passing the scaled residuals
        from the last weighted least squares fit in the IRLS algorithm.

    See Also
    --------
    statsmodels.base.model.LikelihoodModelResults
    """

    def __init__(self, model, params, normalized_cov_params, scale):
        super().__init__(model, params, normalized_cov_params, scale)
        self.model = model
        self.df_model = model.df_model
        self.df_resid = model.df_resid
        self.nobs = model.nobs
        self._cache = {}
        self._data_in_cache.extend(['sresid'])
        self.cov_params_default = self.bcov_scaled

    @cache_readonly
    def fittedvalues(self):
        return np.dot(self.model.exog, self.params)

    @cache_readonly
    def resid(self):
        return self.model.endog - self.fittedvalues

    @cache_readonly
    def sresid(self):
        if self.scale == 0.0:
            sresid = self.resid.copy()
            sresid[:] = 0.0
            return sresid
        return self.resid / self.scale

    @cache_readonly
    def bcov_unscaled(self):
        return self.normalized_cov_params

    @cache_readonly
    def weights(self):
        return self.model.weights

    @cache_readonly
    def bcov_scaled(self):
        model = self.model
        m = np.mean(model.M.psi_deriv(self.sresid))
        var_psiprime = np.var(model.M.psi_deriv(self.sresid))
        k = 1 + (self.df_model + 1) / self.nobs * var_psiprime / m ** 2
        if model.cov == 'H1':
            ss_psi = np.sum(model.M.psi(self.sresid) ** 2)
            s_psi_deriv = np.sum(model.M.psi_deriv(self.sresid))
            return k ** 2 * (1 / self.df_resid * ss_psi * self.scale ** 2) / (1 / self.nobs * s_psi_deriv) ** 2 * model.normalized_cov_params
        else:
            W = np.dot(model.M.psi_deriv(self.sresid) * model.exog.T, model.exog)
            W_inv = np.linalg.inv(W)
            if model.cov == 'H2':
                return k * (1 / self.df_resid) * np.sum(model.M.psi(self.sresid) ** 2) * self.scale ** 2 / (1 / self.nobs * np.sum(model.M.psi_deriv(self.sresid))) * W_inv
            elif model.cov == 'H3':
                return k ** (-1) * 1 / self.df_resid * np.sum(model.M.psi(self.sresid) ** 2) * self.scale ** 2 * np.dot(np.dot(W_inv, np.dot(model.exog.T, model.exog)), W_inv)

    @cache_readonly
    def pvalues(self):
        return stats.norm.sf(np.abs(self.tvalues)) * 2

    @cache_readonly
    def bse(self):
        return np.sqrt(np.diag(self.bcov_scaled))

    @cache_readonly
    def chisq(self):
        return (self.params / self.bse) ** 2

    def summary(self, yname=None, xname=None, title=0, alpha=0.05, return_fmt='text'):
        """
        This is for testing the new summary setup
        """
        top_left = [('Dep. Variable:', None), ('Model:', None), ('Method:', ['IRLS']), ('Norm:', [self.fit_options['norm']]), ('Scale Est.:', [self.fit_options['scale_est']]), ('Cov Type:', [self.fit_options['cov']]), ('Date:', None), ('Time:', None), ('No. Iterations:', ['%d' % self.fit_history['iteration']])]
        top_right = [('No. Observations:', None), ('Df Residuals:', None), ('Df Model:', None)]
        if title is not None:
            title = 'Robust linear Model Regression Results'
        from statsmodels.iolib.summary import Summary
        smry = Summary()
        smry.add_table_2cols(self, gleft=top_left, gright=top_right, yname=yname, xname=xname, title=title)
        smry.add_table_params(self, yname=yname, xname=xname, alpha=alpha, use_t=self.use_t)
        etext = []
        wstr = 'If the model instance has been used for another fit with different fit parameters, then the fit options might not be the correct ones anymore .'
        etext.append(wstr)
        if etext:
            smry.add_extra_txt(etext)
        return smry

    def summary2(self, xname=None, yname=None, title=None, alpha=0.05, float_format='%.4f'):
        """Experimental summary function for regression results

        Parameters
        ----------
        yname : str
            Name of the dependent variable (optional)
        xname : list[str], optional
            Names for the exogenous variables. Default is `var_##` for ## in
            the number of regressors. Must match the number of parameters
            in the model
        title : str, optional
            Title for the top table. If not None, then this replaces the
            default title
        alpha : float
            significance level for the confidence intervals
        float_format : str
            print format for floats in parameters summary

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary2.Summary : class to hold summary results
        """
        from statsmodels.iolib import summary2
        smry = summary2.Summary()
        smry.add_base(results=self, alpha=alpha, float_format=float_format, xname=xname, yname=yname, title=title)
        return smry